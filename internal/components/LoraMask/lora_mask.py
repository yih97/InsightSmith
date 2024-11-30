# Diffusersで動くLoRA。このファイル単独で完結する。
# LoRA module for Diffusers. This file works independently.

import bisect
import logging
import math
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DConditionModel, StableDiffusionControlNetPipeline, ControlNetModel, \
    DPMSolverMultistepScheduler
# this method kohya
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import CLIPTextModel

logger = logging.getLogger(__name__)


def make_unet_conversion_map() -> Dict[str, str]:
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    sd_hf_conversion_map = {sd.replace(".", "_")[:-1]: hf.replace(".", "_")[:-1] for sd, hf in unet_conversion_map}
    return sd_hf_conversion_map


UNET_CONVERSION_MAP = make_unet_conversion_map()


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d" or org_module.__class__.__name__ == "LoRACompatibleConv":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d" or org_module.__class__.__name__ == "LoRACompatibleConv":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)



        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 勾配計算に含めない / not included in gradient calculation

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_forward = org_module.forward
        self.org_module = org_module
        self.enabled = True
        self.mask_dic = None
        self.network: LoRANetwork = None
        self.org_forward = None
        self.mask = None
        self.mask_area = -1

    # override org_module's forward method
    def apply_to(self, multiplier=None):
        self.multiplier=multiplier
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        # del self.org_module


    # restore org_module's forward method
    def unapply_to(self):
        if self.org_forward is not None:
            self.org_module.forward = self.org_forward

    # forward with lora
    # scale is used LoRACompatibleConv, but we ignore it because we have multiplier

    # def to_out_forward(self, x):
    #     # logger.info(f"to_out_forward {self.lora_name} {x.size()} {self.network.is_last_network}")
    #
    #     if self.network.is_last_network:
    #         masks = [None] * self.network.num_sub_prompts
    #         self.network.shared[self.lora_name] = (None, masks)
    #     else:
    #         lx, masks = self.network.shared[self.lora_name]
    #
    #     # call own LoRA
    #     x1 = x[self.network.batch_size + self.network.sub_prompt_index :: self.network.num_sub_prompts]
    #     lx1 = self.lora_up(self.lora_down(x1)) * self.multiplier * self.scale
    #
    #     if self.network.is_last_network:
    #         lx = torch.zeros(
    #             (self.network.num_sub_prompts * self.network.batch_size, *lx1.size()[1:]), device=lx1.device, dtype=lx1.dtype
    #         )
    #         self.network.shared[self.lora_name] = (lx, masks)
    #
    #     # logger.info(f"to_out_forward {lx.size()} {lx1.size()} {self.network.sub_prompt_index} {self.network.num_sub_prompts}")
    #     lx[self.network.sub_prompt_index :: self.network.num_sub_prompts] += lx1
    #     masks[self.network.sub_prompt_index] = self.get_mask_for_x(lx1)
    #
    #     # if not last network, return x and masks
    #     x = self.org_forward(x)
    #     if not self.network.is_last_network:
    #         return x
    #
    #     lx, masks = self.network.shared.pop(self.lora_name)
    #
    #     # if last network, combine separated x with mask weighted sum
    #     has_real_uncond = x.size()[0] // self.network.batch_size == self.network.num_sub_prompts + 2
    #
    #     out = torch.zeros((self.network.batch_size * (3 if has_real_uncond else 2), *x.size()[1:]), device=x.device, dtype=x.dtype)
    #     out[: self.network.batch_size] = x[: self.network.batch_size]  # uncond
    #     if has_real_uncond:
    #         out[-self.network.batch_size :] = x[-self.network.batch_size :]  # real_uncond
    #
    #     # logger.info(f"to_out_forward {self.lora_name} {self.network.sub_prompt_index} {self.network.num_sub_prompts}")
    #     # if num_sub_prompts > num of LoRAs, fill with zero
    #     for i in range(len(masks)):
    #         if masks[i] is None:
    #             masks[i] = torch.zeros_like(masks[0])
    #
    #     mask = torch.cat(masks)
    #     mask_sum = torch.sum(mask, dim=0) + 1e-4
    #     for i in range(self.network.batch_size):
    #         # 1枚の画像ごとに処理する
    #         lx1 = lx[i * self.network.num_sub_prompts : (i + 1) * self.network.num_sub_prompts]
    #         lx1 = lx1 * mask
    #         lx1 = torch.sum(lx1, dim=0)
    #
    #         xi = self.network.batch_size + i * self.network.num_sub_prompts
    #         x1 = x[xi : xi + self.network.num_sub_prompts]
    #         x1 = x1 * mask
    #         x1 = torch.sum(x1, dim=0)
    #         x1 = x1 / mask_sum
    #
    #         x1 = x1 + lx1
    #         out[self.network.batch_size + i] = x1
    #
    #     # logger.info(f"to_out_forward {x.size()} {out.size()} {has_real_uncond}")
    #     return out

    def set_mask_dic(self, mask_dic):
        # called before every generation

        # check this module is related to h,w (not context and time emb)
        if "attn2_to_k" in self.lora_name or "attn2_to_v" in self.lora_name or "emb_layers" in self.lora_name:
            print(f"LoRA for context or time emb: {self.lora_name}")
            self.mask_dic = None

        else:
            self.mask_dic = mask_dic

        self.mask = None

    def forward(self, x, scale=1):
        """
        may be cascaded.
        """
        if self.mask_dic is None:
            # print(self.multiplier, self.scale)
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        # regional LoRA
        # calculate lora and get size
        lx = self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


        mask = self.get_mask_for_x(lx)

        lx = lx * mask

        x = self.org_forward(x)
        x = x + lx

        # if "attn2_to_q" in self.lora_name and self.network.is_last_network:
        #     x = self.postp_to_q(x)
        return x

    # def forward(self, x, scale=1):
    #     self.scale = scale
    #     """
    #     may be cascaded.
    #     """
    #     if self.mask_dic is None:
    #         return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
    #
    #     # regional LoRA
    #
    #     # calculate lora and get size
    #     lx = self.lora_up(self.lora_down(x))
    #
    #     if len(lx.size()) == 4:  # b,c,h,w
    #         area = lx.size()[2] * lx.size()[3]
    #     else:
    #         area = lx.size()[1]  # b,seq,dim
    #
    #     if self.mask is None or self.mask_area != area:
    #         # get mask
    #         # print(self.lora_name, x.size(), lx.size(), area)
    #         mask = self.mask_dic[area]
    #         if len(lx.size()) == 3:
    #             mask = torch.reshape(mask, (1, -1, 1))
    #         self.mask = mask
    #         self.mask_area = area
    #
    #     return self.org_forward(x) + lx * self.multiplier * self.scale * self.mask

    # def postp_to_q(self, x):
    #     self.network.batch_size = 1
    #     # repeat x to num_sub_prompts
    #     has_real_uncond = x.size()[0] // self.network.batch_size == 3
    #     qc = self.network.batch_size  # uncond
    #     qc += self.network.batch_size * self.network.num_sub_prompts  # cond
    #     if has_real_uncond:
    #         qc += self.network.batch_size  # real_uncond
    #
    #     query = torch.zeros((qc, x.size()[1], x.size()[2]), device=x.device, dtype=x.dtype)
    #     query[: self.network.batch_size] = x[: self.network.batch_size]
    #
    #     for i in range(self.network.batch_size):
    #         qi = self.network.batch_size + i * self.network.num_sub_prompts
    #         query[qi : qi + self.network.num_sub_prompts] = x[self.network.batch_size + i]
    #
    #     if has_real_uncond:
    #         query[-self.network.batch_size :] = x[-self.network.batch_size :]
    #
    #     # logger.info(f"postp_to_q {self.lora_name} {x.size()} {query.size()} {self.network.num_sub_prompts}")
    #     return query
    def get_mask_for_x(self, x):
        # calculate size from shape of x
        # self.network.num_sub_prompts = 1
        if len(x.size()) == 4:
            h, w = x.size()[2:4]
            area = h * w
        else:
            area = x.size()[1]
        mask = self.mask_dic[area]
        # mask = self.network.mask_dic.get(area, None)
        if mask is None or len(x.size()) == 2:
            # emb_layers in SDXL doesn't have mask
            # if "emb" not in self.lora_name:
            #     print(f"mask is None for resolution {self.lora_name}, {area}, {x.size()}")
            mask_size = (1, x.size()[1]) if len(x.size()) == 2 else (1, *x.size()[1:-1], 1)
            return torch.ones(mask_size, dtype=x.dtype, device=x.device)
        if len(x.size()) == 3:
            mask = torch.reshape(mask, (1, -1, 1))
        return mask


    def set_network(self, network):
        self.network = network

    # merge lora weight to org weight
    def merge_to(self, multiplier=1.0):
        # get lora weight
        lora_weight = self.get_weight(multiplier)

        # get org weight
        org_sd = self.org_module.state_dict()
        org_weight = org_sd["weight"]
        weight = org_weight + lora_weight.to(org_weight.device, dtype=org_weight.dtype)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module.load_state_dict(org_sd)

    # restore org weight from lora weight
    def restore_from(self, multiplier=1.0):
        # get lora weight
        lora_weight = self.get_weight(multiplier)

        # get org weight
        org_sd = self.org_module.state_dict()
        org_weight = org_sd["weight"]
        weight = org_weight - lora_weight.to(org_weight.device, dtype=org_weight.dtype)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module.load_state_dict(org_sd)

    # return lora weight
    def get_weight(self, multiplier=1):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight
    def set_region(self, region):
        self.region = region
        self.region_mask = None

# Create network from weights for inference, weights are not loaded here
def create_network_from_weights(
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]], unet: UNet2DConditionModel, weights_sd: Dict, multiplier: float = 1.0
):
    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    return LoRANetwork(text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha)


def merge_lora_weights(pipe, weights_sd: Dict, multiplier: float = 1.0):
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2] if hasattr(pipe, "text_encoder_2") else [pipe.text_encoder]
    unet = pipe.unet

    lora_network = create_network_from_weights(text_encoders, unet, weights_sd, multiplier=multiplier)
    lora_network.load_state_dict(weights_sd)
    lora_network.merge_to(multiplier=multiplier)


# block weightや学習に対応しない簡易版 / simple version without block weight and training
class LoRANetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]

    # UNET_TARGET_REPLACE_MODULE = ["SpatialTransformer", "ResBlock", "Downsample", "Upsample"]  # , "Attention"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["ResidualAttentionBlock", "CLIPAttention", "CLIPMLP"]

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet: UNet2DConditionModel,
        multiplier: float = 1.0,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        varbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        logger.info("create LoRA network from weights")


        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],  # None, 1, 2
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )

            loras = []
            skipped = []

            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = (
                            child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "LoRACompatibleLinear"
                        )
                        is_conv2d = (
                            child_module.__class__.__name__ == "Conv2d" or child_module.__class__.__name__ == "LoRACompatibleConv"
                        )

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            if lora_name not in modules_dim:
                                # logger.info(f"skipped {lora_name} (not found in modules_dim)")
                                skipped.append(lora_name)
                                continue

                            dim = modules_dim[lora_name]
                            alpha = modules_alpha[lora_name]
                            lora = LoRAModule(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                            )
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討 / it is wasteful to create all modules every time, need to consider
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
            else:
                index = None

            text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped
        logger.info(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        if len(skipped_te) > 0:
            logger.warning(f"skipped {len(skipped_te)} modules because of missing weight for text encoder.")

        # extend U-Net target modules to include Conv2d 3x3
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE + LoRANetwork.UNET_TARGET_REPLACE_MODULE

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        logger.info(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        if len(skipped_un) > 0:
            logger.warning(f"skipped {len(skipped_un)} modules because of missing weight for U-Net.")
        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            names.add(lora.lora_name)

        for lora_name in modules_dim.keys():
            assert lora_name in names, f"{lora_name} is not found in created LoRA modules."

        # make to work load_state_dict
        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)




    def set_region(self, sub_prompt_index, is_last_network, mask):
        import torch


        if mask.max() == 0:
            mask = torch.ones_like(mask)

        self.mask = mask

        self.sub_prompt_index = sub_prompt_index
        self.is_last_network = is_last_network

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.set_network(self)

    # def set_current_generation(self, width, height, ds_ratio=None):
    #
    #     self.current_size = (height, width)
    #
    #     # create masks
    #     mask = self.mask
    #
    #     mask_dic = {}
    #     mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w
    #     ref_weight = self.text_encoder_loras[0].lora_down.weight if self.text_encoder_loras else self.unet_loras[
    #         0].lora_down.weight
    #     dtype = ref_weight.dtype
    #     device = ref_weight.device
    #     def resize_add(mh, mw):
    #         # logger.info(mh, mw, mh * mw)
    #         m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
    #         m = m.to(device, dtype=dtype)
    #         mask_dic[mh * mw] = m
    #
    #     h = height // 8
    #     w = width // 8
    #     for _ in range(4):
    #         resize_add(h, w)
    #         if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
    #             resize_add(h + h % 2, w + w % 2)
    #
    #         # deep shrink
    #         if ds_ratio is not None:
    #             hd = int(h * ds_ratio)
    #             wd = int(w * ds_ratio)
    #             resize_add(hd, wd)
    #
    #         h = (h + 1) // 2
    #         w = (w + 1) // 2
    #
    #     self.mask_dic = mask_dic

    def set_current_generation(self, width, height, ds_ratio=None):

        self.current_size = (height, width)

        # create masks
        mask = self.mask

        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w
        ref_weight = self.text_encoder_loras[0].lora_down.weight if self.text_encoder_loras else self.unet_loras[
            0].lora_down.weight
        dtype = ref_weight.dtype
        device = ref_weight.device

        def resize_add(mh, mw):
            # logger.info(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(device, dtype=dtype)
            mask_dic[mh * mw] = m

        h = height // 8
        w = width // 8
        for _ in range(4):
            resize_add(h, w)
            if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                resize_add(h + h % 2, w + w % 2)

            # deep shrink
            if ds_ratio is not None:
                hd = int(h * ds_ratio)
                wd = int(w * ds_ratio)
                resize_add(hd, wd)

            h = (h + 1) // 2
            w = (w + 1) // 2

        for lora in self.unet_loras:
            lora.set_mask_dic(mask_dic)
        return

    # SDXL: convert SDXL Stability AI's U-Net modules to Diffusers
    def convert_unet_modules(self, modules_dim, modules_alpha):
        converted_count = 0
        not_converted_count = 0

        map_keys = list(UNET_CONVERSION_MAP.keys())
        map_keys.sort()
        for key in list(modules_dim.keys()):
            if key.startswith(LoRANetwork.LORA_PREFIX_UNET + "_"):
                search_key = key.replace(LoRANetwork.LORA_PREFIX_UNET + "_", "")
                position = bisect.bisect_right(map_keys, search_key)
                map_key = map_keys[position - 1]
                if search_key.startswith(map_key):
                    new_key = key.replace(map_key, UNET_CONVERSION_MAP[map_key])
                    modules_dim[new_key] = modules_dim[key]
                    modules_alpha[new_key] = modules_alpha[key]
                    del modules_dim[key]
                    del modules_alpha[key]
                    converted_count += 1
                else:
                    not_converted_count += 1
        assert (
            converted_count == 0 or not_converted_count == 0
        ), f"some modules are not converted: {converted_count} converted, {not_converted_count} not converted"
        return converted_count

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def apply_to(self, multiplier=1.0, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
            print("enable LoRA for text encoder")
            for lora in self.text_encoder_loras:
                lora.apply_to(multiplier)

        if apply_unet:
            logger.info("enable LoRA for U-Net")
            print("enable LoRA for U-Net")
            for lora in self.unet_loras:
                lora.apply_to(multiplier)

    def unapply_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unapply_to()

    def merge_to(self, multiplier=1.0):
        logger.info("merge LoRA weights to original weights")
        for lora in tqdm(self.text_encoder_loras + self.unet_loras):
            lora.merge_to(multiplier)
        logger.info(f"weights are merged")

    def restore_from(self, multiplier=1.0):
        logger.info("restore LoRA weights from original weights")
        for lora in tqdm(self.text_encoder_loras + self.unet_loras):
            lora.restore_from(multiplier)
        logger.info(f"weights are restored")


if __name__ == "__main__":
    # sample code to use LoRANetwork
    import os
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    import torch
    from datetime import datetime
    from compel import Compel
    import glob
    from pathlib import Path
    from compel import DiffusersTextualInversionManager

    load_dotenv('../../../.env')
    access_token = os.getenv("HF_Token")
    Textual_Inversion_Path=os.getenv("Textual_Inversion_Path")
    model_id = "formsKorea/majicmixrealistic-v7"
    # model_id = "CalypsoCrunchies99/onlyrealistic_v30BakedVAE_fix"
    # lora_weights = ['/home/hamna/Data/m3_10_5/model/m3_10_5-000004.safetensors','/home/hamna/Downloads/only_real_front-000007.safetensors' ]




    image_prefix = model_id.replace("/", "_") + "_"
    image_prefix = os.path.join("/home/hamna/Database/results/lora_mask/", image_prefix)

    # # lie of p setting
    # lora_weights = ['/home/hamna/Data/m3_10_5/model/m3_10_5-000004.safetensors',
    #                '/home/hamna/kyle/Project/p의거짓/p의 거짓 template/lie of p v2-000004.safetensors']
    # prompt = "dels asian man, wearing plg outfit, in plg castle, outdoors, stairs, ascot, mechanical arms, gloves+, magic+, solo, upper body, solo foucs, looking at viewer, front view, standing, portrait+,"
    # negative_prompt = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)1.2, text, cropped, out of frame, (worst quality, low quality)1.2, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers,
    #                     mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs,
    #                     cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,"""
    #
    # mask_path = "/home/hamna/kyle/Project/p의거짓/p의 거짓 template/9c09e65d-16f5-4d62-8454-683a1acff562.png"
    # seed =621590901
    # width = 768
    # height = 768
    # p_canny= Image.open('/home/hamna/kyle/Project/p의거짓/p의 거짓 template/e2687837-8d05-4bc1-aed4-6217daed6080.png')
    # p_open = Image.open('/home/hamna/kyle/Project/p의거짓/p의 거짓 template/a75a0797-36e6-4d9c-93fa-e300b1c564fc.png')
    # p_softeddge = Image.open('/home/hamna/kyle/Project/p의거짓/p의 거짓 template/998608fb-2fdc-4012-b7c7-191a1413e51c.png')
    #
    # controlnet_img = [p_canny,p_open,p_softeddge]

    # battle ground

    lora_weights = ['/home/hamna/Database/lora/hamnatest.safetensors',
                    '/home/hamna/PycharmProjects/stable-diffusion/stable-diffusion-webui/models/Lora/Newspaper.safetensors']

    # lora_weights = ['/home/hamna/kyle/Project/DB/lora/w2_magicmix-000005.safetensors',
    #                 '/home/hamna/kyle/Project/lora_mask_test/bt/battleground-000004.safetensors']

    #4
    prompt = ("portrait of HyEFt, 1girl, newspaper background, full body, looking at viewer, masterpiece, best quality")
    # prompt ="portrait of asian woman wearing btgd helmet in front of btgd box, red smoke, holding gun, weapon, necktie, hat, outdoors, sky, looking at viewer, military, battleground style"
    negative_prompt = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, text, cropped, out of frame, (worst quality, low quality), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers,
                        mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs,
                        cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,"""

    # root_path = '/home/hamna/kyle/Project/lora_mask_test/bt'

    mask_path = '/home/hamna/Database/themes/newspaper/fantasy_1_0_4.png'
    # mask_path = f'{root_path}/btgd_4_mask.png'
    #5
    # seed = 2220346331
    #4
    seed = 2447200372
    width = 768
    height = 1024

    p_canny = Image.open('/home/hamna/Database/themes/newspaper/fantasy_1_0_0.png')
    p_open = Image.open('/home/hamna/Database/themes/newspaper/fantasy_1_0_3.png')
    p_softeddge = Image.open('/home/hamna/Database/themes/newspaper/fantasy_1_0_2.png')


    # gt setting
    # lora_weights = ['/home/hamna/kyle/Project/DB/lora/w2_magicmix-000005.safetensors',
    #                 '/home/hamna/kyle/Project/lora_mask_test/gy/Gyeongbokgung all-000004.safetensors']
    #
    # prompt = "dels, photo of dels asian woman in hko street, looking at viewer, upper body, sitting, holding skirt, white shirt, red skirt, floral print, outdoors, east asian architecture, hanbok"
    # negative_prompt = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)1.2, text, cropped, out of frame, (worst quality, low quality)1.2, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers,
    #                     mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs,
    #                     cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,"""
    #
    # root_path = '/home/hamna/kyle/Project/lora_mask_test/gy'
    #
    # mask_path = f'{root_path}/street2_m.png'
    #
    # seed = 719312933
    # width = 512
    # height = 512
    # p_canny = Image.open(f'{root_path}/street2_c.png')
    #
    # p_open = Image.open(f'{root_path}/street2_o.png')
    # p_softeddge = Image.open(f'{root_path}/street2_s.png')



    controlnet_img = [p_canny, p_open, p_softeddge]

    controlnet_openpose = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose',torch_dtype=torch.float16)
    controlnet_softedge = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge',torch_dtype=torch.float16)
    controlnet_canny = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny',torch_dtype=torch.float16)




    pipe: Union[StableDiffusionPipeline, StableDiffusionXLPipeline]
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id,controlnet=[controlnet_canny,controlnet_openpose,controlnet_softedge],
                                                             token=access_token, torch_dtype=torch.float16)

    pipe.to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_use_memory_efficient_attention_xformers(True)


    text_encoders = [pipe.text_encoder]


    # 必要があれば、元のモデルの重みをバックアップしておく
    # back-up unet/text encoder weights if necessary
    def detach_and_move_to_cpu(state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu()
        return state_dict

    org_unet_sd = pipe.unet.state_dict()
    detach_and_move_to_cpu(org_unet_sd)

    org_text_encoder_sd = pipe.text_encoder.state_dict()
    detach_and_move_to_cpu(org_text_encoder_sd)



    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # create image with original weights

    logger.info(f"create image with original weights")


    text_inversion_path_list = glob.glob(os.path.join(Textual_Inversion_Path, '*.pt'))
    token = []

    for text_inversion in text_inversion_path_list:
        text_inversion_name = Path(text_inversion).stem
        token.append(text_inversion_name)

    pipe.load_textual_inversion(text_inversion_path_list, token=token)

    textual_inversion_manager = DiffusersTextualInversionManager(pipe)

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                    textual_inversion_manager=textual_inversion_manager)
    prompt_embeds = compel(prompt)
    negative_prompt_embeds = compel(negative_prompt)


    networks = []
    for lora_weight in lora_weights:
        logger.info(f"load LoRA weights from {lora_weight}")
        if os.path.splitext(lora_weight)[1] == ".safetensors":
            from safetensors.torch import load_file

            lora_sd = load_file(lora_weight)
        else:
            lora_sd = torch.load(lora_weight)

        # create by LoRA weights and load weights
        logger.info(f"create LoRA network")
        network: LoRANetwork = create_network_from_weights(text_encoders, pipe.unet, lora_sd, multiplier=1)

        logger.info(f"load LoRA network weights")

        network.load_state_dict(lora_sd, False)

        network.to("cuda", dtype=pipe.unet.dtype)  # required to apply_to. merge_to works without this
        # apply LoRA network to the model: slower than merge_to, but can be reverted easily
        logger.info(f"apply LoRA network to the model")
        network.apply_to(multiplier=0.8)


        networks.append(network)

    if networks and mask_path:
        # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
        regional_network = True
        logger.info("use mask as region")
        mask_image = [Image.open(mask_path).convert("RGB")]
        size = None
        logger.info("use mask as region")

        for i, network in enumerate(networks):
            np_mask = np.array(mask_image[0])
            if i == 0:
                ch0 = 1
                ch1 = 0
                ch2 = 0

            elif i == 1:
                ch1 = 1
                ch0 = 0
                ch2 =0

            else:
                ch0 = 0
                ch1 = 0
                ch2 = 1


            np_mask = np.all(np_mask >= np.array([ch0, ch1, ch2]) * 230, axis=2)

            np_mask = np_mask.astype(np.uint8) * 255

            size = np_mask.shape

            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)

            network.set_region(i, i == len(networks) - 1, mask)

            print(f"apply mask, channel: {['R', 'G', 'B'][i]}, model: {lora_weights[i]}")
        mask_images = None

        for n in networks:
            n.set_current_generation(width=width,height=height)

        logger.info(f"create image with applied LoRA")
        seed_everything(seed)

        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                     num_inference_steps=35,
                guidance_scale=7,
                clip_skip=2,
                controlnet_conditioning_scale=[0.8,0.8,0.8],
                height=height,
                width=width,
                image=controlnet_img,
                ).images[0]

        # print("print('1') 실행 횟수:", print_count1)
        # print("print('2') 실행 횟수:", print_count2)

        id_ = datetime.now().strftime("%Y%m%d%H%M%S")
        image.save(image_prefix + f"applied_lora_{id_}.png")
        torch.cuda.empty_cache()