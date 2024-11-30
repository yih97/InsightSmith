from __future__ import annotations
import torch
import os
import logging
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import random
import json
from diffusers import StableDiffusionControlNetPipeline
from internal.components.LoraMask.lora_mask import LoRANetwork, create_network_from_weights
from internal.components.Adetailer import AdCnPipeline


def seed_everything(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def use_lora_mask(pipeline: AdCnPipeline | StableDiffusionControlNetPipeline,
                  lora_paths: list[str],
                  lora_scale: list[float], generation_settings: json):

    """Activate the LORA mask.
    Args:pipeline: Pipeline to be used for image generation
    lora_paths (list): List of paths to the LORA weights file
    lora_scale (list): List of scaling factors for the LORA weights
    generation_settings (json): Generation settings for the images

    Returns:
    list[LoRANetwork]: List of LoRANetwork objects for the specified LORA weights
        """
    pipe = pipeline
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


    # create image with original weights

    logging.info(f"create image with original weights")

    networks = []
    for i, lora_path in enumerate(lora_paths):
        logging.info(f"load LoRA weights from {lora_path}")
        if os.path.splitext(lora_path)[1] == ".safetensors":
            from safetensors.torch import load_file

            lora_sd = load_file(lora_path)
        else:
            lora_sd = torch.load(lora_path)

        # create by LoRA weights and load weights
        logging.info(f"create LoRA network")
        network: LoRANetwork = create_network_from_weights(text_encoders, pipe.unet, lora_sd, multiplier=lora_scale[i])

        logging.info(f"load LoRA network weights")

        network.load_state_dict(lora_sd, False)

        network.to("cuda", dtype=pipe.unet.dtype)  # required to apply_to. merge_to works without this
        # apply LoRA network to the model: slower than merge_to, but can be reverted easily
        logging.info(f"apply LoRA network to the model")
        network.apply_to(multiplier=lora_scale[i])
        networks.append(network)

    if networks and generation_settings['loramask']:
        # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
        regional_network = True
        logging.info("use mask as region")
        mask_image = [Image.open(generation_settings['loramask']).convert("RGB")]
        logging.info("use mask as region")

        for i, network in enumerate(networks):
            np_mask = np.array(mask_image[0])
            if i == 0:
                ch0 = 1
                ch1 = 0
                ch2 = 0

            elif i == 1:
                ch0 = 0
                ch1 = 1
                ch2 = 0

            else:
                ch0 = 0
                ch1 = 0
                ch2 = 1

            np_mask = np.all(np_mask >= np.array([ch0, ch1, ch2]) * 230, axis=2)

            np_mask = np_mask.astype(np.uint8) * 255

            size = np_mask.shape

            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)

            network.set_region(i, i == len(networks) - 1, mask)

            logging.info(f"apply mask, channel: {['R', 'G', 'B'][i]}, model: {lora_paths[i]}")
            print(f"apply mask, channel: {['R', 'G', 'B'][i]}, model: {lora_paths[i]}")

        # mask_image = None

        for n in networks:
            n.set_current_generation(width=generation_settings['width'], height=generation_settings['height'])

        logging.info(f"create image with applied LoRA")
        seed_everything(generation_settings['seed'])

        return networks
