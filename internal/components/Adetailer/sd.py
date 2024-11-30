from __future__ import annotations

from functools import cached_property

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

from .base import *

# class AdInpaintCnPipeline(AdPipelineBase, StableDiffusionControlNetInpaintPipeline):
#     @property
#     def txt2img_class(self):
#         return StableDiffusionControlNetInpaintPipeline
#
#     @property
#     def inpaint_pipeline(self):
#         return StableDiffusionInpaintPipeline(
#             vae=self.vae,
#             text_encoder=self.text_encoder,
#             tokenizer=self.tokenizer,
#             unet=self.unet,
#             scheduler=self.scheduler,
#             safety_checker=self.safety_checker,
#             feature_extractor=self.feature_extractor,
#             requires_safety_checker=self.config.requires_safety_checker,
#         )

class AdInpaintCnPipeline(AdPipelineBase, StableDiffusionControlNetInpaintPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet, scheduler, safety_checker, feature_extractor,
                         image_encoder, requires_safety_checker)
        self.pipe = None

    @cached_property
    def inpaint_pipeline(self):
        print("Inpaint")
        # return StableDiffusionControlNetInpaintPipeline(
        #     vae=self.vae,
        #     text_encoder=self.text_encoder,
        #     tokenizer=self.tokenizer,
        #     unet=self.unet,
        #     controlnet=self.controlnet,
        #     scheduler=self.scheduler,
        #     safety_checker=self.safety_checker,
        #     feature_extractor=self.feature_extractor,
        #     image_encoder=self.image_encoder,
        #     requires_safety_checker=self.config.requires_safety_checker,
        # )
        # return StableDiffusionInpaintPipeline.from_pretrained('formsKorea/majicmixrealistic-v7',
        #                                                       token='').to(
        #     'cuda')
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
                )

    @property
    def txt2img_class(self):
        print("Txt2Img")
        self.pipe = StableDiffusionControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            image_encoder=self.image_encoder,
            requires_safety_checker=self.config.requires_safety_checker,
        )
        return self.pipe



class AdPipeline(AdPipelineBase, StableDiffusionPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionPipeline

class AdInpaintPipeline(AdPipelineBase, StableDiffusionControlNetPipeline):
    @property
    def txt2img_class(self, **kwargs):
        return StableDiffusionControlNetPipeline

    @property
    def inpaint_pipeline(self):

        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )



class AdCnPipeline(AdPipelineBase, StableDiffusionControlNetPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet, scheduler, safety_checker, feature_extractor,
                         image_encoder, requires_safety_checker)
        self.pipe = None

    @cached_property
    def inpaint_pipeline(self):
        print("Inpaint")
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
                )

    @property
    def txt2img_class(self):
        print("Txt2Img")
        self.pipe = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            image_encoder=self.image_encoder,
            requires_safety_checker=self.config.requires_safety_checker,
        )
        return self.pipe