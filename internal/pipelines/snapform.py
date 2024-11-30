# Version: 1.0
"""
This script runs the snapform pipeline to generate images using the specified theme and settings.
The pipeline uses controlnet and stable-diffusion to generate images.
"""

import os
import glob
import json
from typing import Any
from dotenv import dotenv_values
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, AutoencoderKL
from compel import Compel
from internal.components.Adetailer import AdCnPipeline, mediapipe_face_mesh_eyes_only
from internal.components.LoraMask import use_lora_mask
from internal.components.yolo import yolo_detector
from internal.metrics.metrics_functions import *
import logging
from datetime import datetime
from compel import DiffusersTextualInversionManager
from pathlib import Path
from modules.service.tinydb_mangement import exist_ti
from internal.components import Adetailer


##Snapform Version#1

class SnapForm:
    """

    Pipeline to generate images using snapform pipeline.
    Attributes:
        config (dict): Configuration settings read from the .env file
        controlnet_openpose (ControlNetModel): ControlNet model for Openpose
        controlnet_canny (ControlNetModel): ControlNet model for Canny edge detection
        controlnet_softedge (ControlNetModel): ControlNet model for soft edge detection
        checkpoint (StableDiffusionControlNetPipeline): Stable Diffusion ControlNet pipeline for image generation
        initialised using specified model
        adetailer (Adetailer.AdPipeline): Adetailer pipeline for enhancing the generated images
        compel (Compel): Compel object for text embedding

    """

    # pylint: disable=too-many-instance-attributes
    # Twelve is reasonable in this case.
    # pylint: disable=too-many-locals
    # Twenty Seven is reasonable in this case.

    def __init__(self, adetailer=True):
        """
        Initialize the snapform pipeline. Reads the configuration from the .env file and initializes the required
        models and components.

        Args: adetailer (bool): Flag to determine if Adetailer is to be used for enhancing the generated images.
        Default is false.

        """
        self.networks = []
        self.config = dotenv_values(".env")
        self.use_adetailer = adetailer
        self.controlnet_openpose = ControlNetModel.from_pretrained(self.config['CONTROLNET_OPENPOSE'])
        self.controlnet_canny = ControlNetModel.from_pretrained(self.config['CONTROLNET_CANNY'])
        self.controlnet_softedge = ControlNetModel.from_pretrained(self.config['CONTROLNET_SOFTEDGE'])

        self.lora_weight = 0.5
        if not self.use_adetailer:
            self.checkpoint = StableDiffusionControlNetPipeline.from_pretrained(self.config['CHECKPOINT'],
                                                                                controlnet=[self.controlnet_openpose,
                                                                                            self.controlnet_canny,
                                                                                            self.controlnet_softedge],

                                                                                # use_fp16=True,
                                                                                # torch_dtype=torch.float16,
                                                                                token=self.config['HF_TOKEN']).to(
                'cuda')
            vae = AutoencoderKL.from_pretrained(self.config['VAE']).to('cuda')
            self.checkpoint.vae = vae
            self.checkpoint.scheduler = DPMSolverMultistepScheduler.from_config(self.checkpoint.scheduler.config)
            self.checkpoint.enable_xformers_memory_efficient_attention()

        else:

            self.checkpoint = AdCnPipeline.from_pretrained(self.config['CHECKPOINT'],
                                                           controlnet=[self.controlnet_openpose,
                                                                       self.controlnet_canny,
                                                                       self.controlnet_softedge],
                                                           # use_fp16=True,
                                                           # torch_dtype=torch.float16,
                                                           token=self.config['HF_TOKEN']).to('cuda')
            vae = AutoencoderKL.from_pretrained(self.config['VAE']).to('cuda')
            self.checkpoint.vae = vae
            self.checkpoint.scheduler = DPMSolverMultistepScheduler.from_config(self.checkpoint.scheduler.config)
            self.checkpoint.enable_xformers_memory_efficient_attention()

            self.default_unet = self.checkpoint.unet
            self.default_text_encoder = self.checkpoint.text_encoder

        self.compel = Compel(tokenizer=self.checkpoint.tokenizer,
                             text_encoder=self.checkpoint.text_encoder)

    def load_lora(self, lora_path):
        """Load the LORA weights to the pipeline.
        Args:
        lora_path (str): Path to the LORA weights file
        """
        self.checkpoint.load_lora_weights(lora_path)

    #     def use_lora_mask(self, pipeline, lora_paths: list[str], lora_scale: list[float], generation_settings: json):
    #         """Activate the LORA mask.
    #         Args:pipeline: Pipeline to be used for image generation
    #         lora_paths (list): List of paths to the LORA weights file
    #         lora_scale (list): List of scaling factors for the LORA weights
    #         generation_settings (json): Generation settings for the images

    #         Returns:
    #         list[LoRANetwork]: List of LoRANetwork objects for the specified LORA weights
    #             """
    #         pipe = pipeline
    #         text_encoders = [pipe.text_encoder]

    #         # 必要があれば、元のモデルの重みをバックアップしておく
    #         # back-up unet/text encoder weights if necessary
    #         def detach_and_move_to_cpu(state_dict):
    #             for k, v in state_dict.items():
    #                 state_dict[k] = v.detach().cpu()
    #             return state_dict

    #         org_unet_sd = pipe.unet.state_dict()
    #         detach_and_move_to_cpu(org_unet_sd)

    #         org_text_encoder_sd = pipe.text_encoder.state_dict()
    #         detach_and_move_to_cpu(org_text_encoder_sd)

    #         def seed_everything(seed):
    #             if seed == -1:
    #                 seed = random.randint(0, 1000000)
    #             torch.manual_seed(seed)
    #             torch.cuda.manual_seed_all(seed)
    #             np.random.seed(seed)
    #             random.seed(seed)

    #         # create image with original weights

    #         logging.info("create image with original weights")
    #         networks = []
    #         for i, lora_path in enumerate(lora_paths):
    #             logging.info("load LoRA weights from %s", lora_path)
    #             if os.path.splitext(lora_path)[1] == ".safetensors":

    #                 lora_sd = load_file(lora_path)
    #             else:
    #                 lora_sd = torch.load(lora_path)

    #             # create by LoRA weights and load weights
    #             logging.info("create LoRA network")
    #             network: LoRANetwork = create_network_from_weights(text_encoders, pipe.unet, lora_sd,
    #                                                                multiplier=lora_scale[i])

    #             logging.info("load LoRA network weights")

    #             network.load_state_dict(lora_sd, False)

    #             network.to("cuda", dtype=pipe.unet.dtype)  # required to apply_to. merge_to works without this
    #             # apply LoRA network to the model: slower than merge_to, but can be reverted easily
    #             logging.info("apply LoRA network to the model")
    #             network.apply_to(multiplier=lora_scale[i])
    #             networks.append(network)

    #         if networks and generation_settings['loramask']:
    #             # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
    #             regional_network = True
    #             logging.info("use mask as region")
    #             mask_image = [Image.open(generation_settings['loramask']).convert("RGB")]
    #             logging.info("use mask as region")

    #             for i, network in enumerate(networks):
    #                 np_mask = np.array(mask_image[0])
    #                 if i == 0:
    #                     ch0 = 1
    #                     ch1 = 0
    #                     ch2 = 0

    #                 elif i == 1:
    #                     ch0 = 0
    #                     ch1 = 1
    #                     ch2 = 0

    #                 else:
    #                     ch0 = 0
    #                     ch1 = 0
    #                     ch2 = 1

    #                 np_mask = np.all(np_mask >= np.array([ch0, ch1, ch2]) * 230, axis=2)

    #                 np_mask = np_mask.astype(np.uint8) * 255

    #                 size = np_mask.shape

    #                 mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)

    #                 network.set_region(i, i == len(networks) - 1, mask)

    #                 logging.info(f"apply mask, channel: {['R', 'G', 'B'][i]}, model: {lora_paths[i]}")
    #                 print(f"apply mask, channel: {['R', 'G', 'B'][i]}, model: {lora_paths[i]}")
    #             mask_images = None

    #             for n in networks:
    #                 n.set_current_generation(width=generation_settings['width'], height=generation_settings['height'])

    #             logging.info("create image with applied LoRA")
    #             # seed_everything(generation_settings['seed'])

    #             return networks

    def load_textual_inversions(self):
        """Load the textual inversion weights to the pipeline.
        """
        try:
            text_inversion_path_list = glob.glob(os.path.join(self.config['Textual_Inversion_Path'], '*.pt'))
            if len(text_inversion_path_list) == 0:
                text_inversion_path_list = exist_ti()
            token = []
            for text_inversion in text_inversion_path_list:
                text_inversion_name = Path(text_inversion).stem
                token.append(text_inversion_name)
                self.checkpoint.load_textual_inversion(text_inversion, token=token)

            textual_inversion_manager = DiffusersTextualInversionManager(self.checkpoint)
            self.compel = Compel(tokenizer=self.checkpoint.tokenizer, text_encoder=self.checkpoint.text_encoder,
                                 textual_inversion_manager=textual_inversion_manager)
            logging.info("Textual inversion loaded successfully")
        except Exception as e:
            logging.error("Error while loading textual inversion: %s", e)

        return

    def reset_variables(self):
        """
        Reset the variables to their default values.
        """
        default_values = {
            'networks': [],
            'config': dotenv_values(".env"),
            'use_adetailer': self.use_adetailer,
            'controlnet_openpose': ControlNetModel.from_pretrained(self.config['Controlnet_Openpose']),
            'controlnet_canny': ControlNetModel.from_pretrained(self.config['Controlnet_Canny']),
            'controlnet_softedge': ControlNetModel.from_pretrained(self.config['Controlnet_Softedge']),
            'checkpoint': None,
            'compel': None,
        }

        if not self.use_adetailer:
            default_values['checkpoint'] = StableDiffusionControlNetPipeline.from_pretrained(
                self.config['Checkpoint'],
                controlnet=[default_values['controlnet_openpose'], default_values['controlnet_canny'],
                            default_values['controlnet_softedge']],
                token=self.config['HF_Token']
            ).to('cuda')
            vae = AutoencoderKL.from_pretrained(self.config['VAE']).to('cuda')
            default_values['checkpoint'].vae = vae
            default_values['checkpoint'].scheduler = DPMSolverMultistepScheduler.from_config(
                default_values['checkpoint'].scheduler.config)
            default_values['checkpoint'].enable_xformers_memory_efficient_attention()
        else:
            default_values['checkpoint'] = AdCnPipeline.from_pretrained(
                self.config['Checkpoint'],
                controlnet=[default_values['controlnet_openpose'], default_values['controlnet_canny'],
                            default_values['controlnet_softedge']],
                token=self.config['HF_Token']
            ).to('cuda')
            vae = AutoencoderKL.from_pretrained(self.config['VAE']).to('cuda')
            default_values['checkpoint'].vae = vae
            default_values['checkpoint'].scheduler = DPMSolverMultistepScheduler.from_config(
                default_values['checkpoint'].scheduler.config)
            default_values['checkpoint'].enable_xformers_memory_efficient_attention()

        default_values['compel'] = Compel(
            tokenizer=default_values['checkpoint'].tokenizer,
            text_encoder=default_values['checkpoint'].text_encoder
        )

        current_vars = vars(self)
        for var in current_vars:
            if var in default_values:
                setattr(self, var, default_values[var])
            else:
                setattr(self, var, None)

    def __call__(self,
                 lora_id: str,
                 gender: str,
                 token: str,
                 generation_settings: json,
                 ) -> tuple[Any, None, None] | tuple[Any, Any, Any] | list[Any]:
        """
        Generate images using the snapform pipeline.

        Args:
        canny_template (Image.Image): Canny edge detection template. Preprocessed and ready for usage
        openpose_template (Image.Image): Openpose template. Preprocessed and ready for usage
        softedge_template (Image.Image): Soft edge detection template. Preprocessed and ready for usage
        lora_id (str): Lora ID of the user. Will be used to create the users face
        generation_settings (json): Generation settings for the images. Will vary based
        on the template used.

        Returns:
        tuple[Image.Image, Image.Image]: Tuple containing the generated images and the enhanced images in
        PIL format. The generated images are the images generated using diffusion based on settings. The
        enhanced images are the generated images enhanced using Adetailer.

        #letting function use json as genrationsettings so if needed we can let user pass all the parameters instead of
        just theme_id
        """
        if gender == 'f':
            self.lora_weight = self.config['LORA_WEIGHT_WOMAN']
        elif gender == 'm':
            self.lora_weight = self.config['LORA_WEIGHT_MAN']
        else:
            self.lora_weight = 0.5

        # load lora to SD pipeline
        lora_path = os.path.join(self.config['LORA_PATH'], lora_id + '.safetensors')
        if generation_settings['use_lora_mask']:
            generation_settings['template_lora_path'].append(lora_path)
            lora_paths = generation_settings['template_lora_path']
            generation_settings['template_lora_weights'].append(float(self.lora_weight))
            lora_weights = generation_settings['template_lora_weights']
            if isinstance(self.checkpoint, AdCnPipeline):
                #Load lora mask to txt2img and inpaint pipeline
                #To-do: test if adding lora mask to inpaint pipeline is necessary or not
                # pipe = [self.checkpoint, self.checkpoint.inpaint_pipeline]
                # for p in pipe:
                #     lora_networks = use_lora_mask(pipeline=p, lora_paths=lora_paths, lora_scale=lora_weights,
                #                                   generation_settings=generation_settings)
                #     self.networks.append(lora_networks)
                #Load lora mask to txt2img pipeline only
                self.networks = use_lora_mask(pipeline=self.checkpoint, lora_paths=lora_paths, lora_scale=lora_weights,
                                              generation_settings=generation_settings)

        else:
            self.load_lora(lora_path)

        #load textual inversion
        #To-do: test if loading textual inversion is necessary or not
        self.load_textual_inversions()

        # Prepare controlnet component
        controlnet_images = []
        generated_images = []

        # Open each control net image and append it to the list
        for path in generation_settings['controlnet_images']:
            img = Image.open(path)
            controlnet_images.append(img)

        # fix the seed value
        generator = torch.Generator(device='cuda')
        generator.manual_seed(generation_settings['seed'])

        #Add token to the positive prompt
        generation_settings['positive_prompt'] = token + generation_settings['positive_prompt']

        if generation_settings['use_compel']:
            prompt_embeds = self.compel(generation_settings['positive_prompt'])
            negative_prompt_embeds = self.compel(generation_settings['negative_prompt'])

            if not self.use_adetailer:

                generated_images = self.generate_embeds(prompt_embeds, negative_prompt_embeds,
                                                        generation_settings,
                                                        controlnet_images, generator)
            else:
                generated_images = self.adetailer_generate(prompt_embeds, negative_prompt_embeds,
                                                           generation_settings,
                                                           controlnet_images, generator)
        else:
            if not self.use_adetailer:
                generated_images = self.generate_prompts(generation_settings, controlnet_images, generator)
            else:
                logging.error("Adetailer cannot be used without Compel embeddings")

        self.remove_lora()
        torch.cuda.empty_cache()
        return generated_images

    def remove_lora(self):
        """
        Remove the LORA weights from the pipeline.
        """
        print("before:", len(self.networks))
        for lora_network in self.networks:
            lora_network.unapply_to()
            self.networks.remove(lora_network)
        print("after:", len(self.networks))
        self.checkpoint.unet = self.default_unet
        self.checkpoint.text_encoder = self.default_text_encoder

    def generate_prompts(self, generation_settings,
                         controlnet_images, generator) -> tuple[Any, None, None]:
        generated_images = self.checkpoint(
            prompt=generation_settings['positive_prompt'],
            negative_prompt=generation_settings['negative_prompt'],
            image=controlnet_images,
            num_inference_steps=generation_settings['num_inference_steps'],
            strength=generation_settings['strength'],
            clip_skip=generation_settings['clip_skip'],
            num_images_per_prompt=generation_settings['num_images_per_prompt'],
            generator=generator
        ).images
        self.checkpoint.unload_lora_weights()

        return generated_images, None, None

    def generate_embeds(self, prompt_embeds, negative_prompt_embeds, generation_settings,
                        controlnet_images, generator) -> tuple[Any, None, None]:
        generated_images = self.checkpoint(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=controlnet_images,
            num_inference_steps=generation_settings['num_inference_steps'],
            guidance_scale=generation_settings['guidance_scale'],
            clip_skip=generation_settings['clip_skip'],
            num_images_per_prompt=generation_settings['num_images_per_prompt'],
            generator=generator
        ).images
        self.checkpoint.unload_lora_weights()

        return generated_images, None, None

    def adetailer_generate(self, prompt_embeds, negative_prompt_embeds, generation_settings,
                           controlnet_images, generator) -> tuple[Any, Any, Any]:

        common = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "num_inference_steps": generation_settings['num_inference_steps'],
            "image": controlnet_images,
            "width": generation_settings['width'],
            "height": generation_settings['height'],
            "generator": generator,
            "controlnet_conditioning_scale": generation_settings['controlnet_conditioning_scale'],
            "clip_skip": generation_settings['clip_skip'],
            "num_images_per_prompt": generation_settings['num_images_per_prompt']
            #Used to load lora weights
            # "cross_attention_kwargs": {
            #     "scale": float(self.config['Lora_Weight_Woman'])
            # }

            # "strength": generation_settings['strength']

        }
        enhanced_images = self.checkpoint(common=common,
                                          txt2img_only={"guidance_scale": generation_settings['guidance_scale'],
                                                        },
                                          inpaint_only={"strength": 0.3, "guidance_scale": 3},
                                          detectors=[yolo_detector, mediapipe_face_mesh_eyes_only])
        self.remove_lora()

        return enhanced_images.images, enhanced_images.init_images, enhanced_images.inpaint_images

    def generate_id(self):
        """
        Generate a unique id for the images based on the current timestamp.
        :returns: String containing the unique id
        """
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def save_images(self, generated_images: List[Image.Image], enhanced_images: List[Image.Image]):
        """
        Save the generated and enhanced images to the specified path.

        Args:
        generated_images (List[Image.Image]): List of generated images to be saved
        enhanced_images (List[Image.Image]): List of enhanced images to be saved
        path (str): Path where the images will be saved

        Returns:
        None
        """
        id_ = self.generate_id()
        for i, (generated, enhanced) in enumerate(zip(generated_images, enhanced_images)):
            generated.save(os.path.join(self.config['SAVE_PATH_GEN'], f'{id_}_generated_{i}.png'))
            enhanced.save(os.path.join(self.config['SAVE_PATH_ENHANCED'], f'{id_}_enhanced_{i}.png'))
            logging.info(f"Images saved at {self.config['SAVE_PATH_GEN']} and "
                         f"{self.config['SAVE_PATH_ENHANCED']} with id {id_}")

    def save_images_(self, generated_images: List[Image.Image]):
        """
        Save the generated and enhanced images to the specified path.

        Args:
        generated_images (List[Image.Image]): List of generated images to be saved
        enhanced_images (List[Image.Image]): List of enhanced images to be saved
        path (str): Path where the images will be saved

        Returns:
        None
        """
        for i, enhanced in enumerate(generated_images):
            id_ = datetime.now().strftime("%Y%m%d%H%M%S")
            enhanced.save(os.path.join(self.config['TEST_PATH'], f'{id_}_generateded_{i}.png'))
            logging.info(f"Images saved at {self.config['TEST_PATH']} with id {id_}")

# if __name__ == '__main__':
#     import os
#     import glob
#     from internal.components.utils import load_and_check_json_files, modify_controlnet_images
#     from internal.logger import *
#     from internal.components.face_processing import face_recognition
#     from internal.metrics.metrics_functions import *


#     """Run this line of code once to format the controlnet images in the required format"""
#     # modify_controlnet_images("/home/hamna/Database/Tem/", "/home/hamna/Database/Tem")

#     snapform = SnapForm(adetailer=True)
#     templates = load_and_check_json_files(snapform.config['THEME_PATHS'])
#     gender = 'F'
#     # user = Image.open('/home/lh/Documents/Database/user/7.jpg')
#     # log = logger()
#     # log.initialize()
#     # log.create_table("test", ["user", "enhanced", "generated"])

#     for template in templates:
#         print(template)
#         with open(template) as json_file:
#             settings = json.load(json_file)

#         enhanced, generated, inpaint = snapform('w2', gender, 'dels', settings)

#         snapform.save_images(generated, enhanced)
#         snapform.save_images_(inpaint)

#         # score = face_recognition(enhanced, user)

#         # ssim_scores = [calculate_ssim(user, enhanced)
#         #                for user, enhanced in zip(user, [enhanced])]

#         # log.log_data({
#         #     'user': wandb.Image(user),
#         #     'enhanced': wandb.Image(enhanced[0]),
#         #     'generate': wandb.Image(generated[0]),
#         #     'score': score[0]['distance'],
#         #     'ssim_score': ssim_scores[0],
#         # })
#     print(len(generated))
#                          f"{self.config['test_path']} with id {id_}")

# if __name__ == '__main__':
#     import os
#     import glob
#     from internal.components.utils import load_and_check_json_files, modify_controlnet_images
#     from internal.logger import Logger
#     from internal.components.face_processing import face_recognition
#
#     """Run this line of code once to format the controlnet images in the required format"""
#     # modify_controlnet_images("/home/hamna/Database/Tem/", "/home/hamna/Database/Tem")
#
#     snapform = SnapForm(adetailer=True)
#     templates = load_and_check_json_files(snapform.config['Theme_Paths'])
#     gender = 'F'
#     user = Image.open('/media/hamna/Extra/dataset/People/w1/data/IMG_1786.jpg')
#     log = Logger()
#     log.initialize()
#     log.create_table("test", ["user", "enhanced", "generated"])
#
#     for template in templates:
#         print(template)
#         with open(template) as json_file:
#             settings = json.load(json_file)
#
#         enhanced, generated, inpaint = snapform('w1', gender, settings)
#
#         snapform.save_images(generated, enhanced)
#         snapform.save_images_(inpaint)
#
#         score = face_recognition(enhanced, user)
#
#         log.log_data({
#             'user': wandb.Image(user),
#             'enhanced': wandb.Image(enhanced[0]),
#             'generate': wandb.Image(generated[0]),
#             'score': score[0]['distance'],
#         })
#     print(len(generated))
