import io
import json

import tinydb.table as table
from tinydb import TinyDB

from internal.components.Adetailer import AdCnPipeline, mediapipe_face_mesh_eyes_only, AdInpaintCnPipeline
from PIL import Image, ImageFilter
from dotenv import dotenv_values
from diffusers import ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler
from internal.components.yolo import yolo_instance_segmentation, yolo_detector
import numpy as np
import cv2
import logging
from modules.s3 import S3Service
from modules.formsGenerator.generator_instances import GenerateOptions, FaceLoraOptions
import os
from compel import DiffusersTextualInversionManager
from compel import Compel
from pydantic import BaseModel
from internal.components.edge_detection import calculate_canny, calculate_softedge
from internal.components.openpose import OpenposeDetector
import torch
import glob
from pathlib import Path

from typing import Any, Tuple

# OpenposeDetector = OpenposeDetector()


class FormsGenerator:
    def __init__(self, use_erode=True, generated_width=768, generated_height=1024):
        """
        Initialize the forms generator pipeline.
        Args:
        use_erode (bool): Flag to enable eroding the boundaries of the image. (default is True)
        generated_width (int): Width of the generated image. (default is 768)
        generated_height (int): Height of the generated image. (default is 1024)

        """
        self.config = dotenv_values()
        self.controlnet_openpose = ControlNetModel.from_pretrained(self.config['Controlnet_Openpose'])
        self.controlnet_canny = ControlNetModel.from_pretrained(self.config['Controlnet_Canny'])
        self.controlnet_softedge = ControlNetModel.from_pretrained(self.config['Controlnet_Softedge'])
        self.checkpoint = AdInpaintCnPipeline.from_pretrained(self.config['Checkpoint'],
                                                              controlnet=[self.controlnet_openpose,
                                                                          self.controlnet_softedge,
                                                                          self.controlnet_canny],
                                                              # use_fp16=True,
                                                              # torch_dtype=torch.float16,
                                                              token=self.config['HF_Token']).to('cuda')
        vae = AutoencoderKL.from_pretrained(self.config['VAE']).to('cuda')
        self.checkpoint.vae = vae
        self.checkpoint.scheduler = DPMSolverMultistepScheduler.from_config(self.checkpoint.scheduler.config)
        self.checkpoint.enable_xformers_memory_efficient_attention()

        self.generated_width = generated_width
        self.generated_height = generated_height
        self.use_erode = use_erode
        self.compel = Compel(tokenizer=self.checkpoint.tokenizer,
                             text_encoder=self.checkpoint.text_encoder)

    def get_controlnet_imgs(self, image: Image):
        """
        Get the controlnet conditioning image from the input image.
        Args:
            image (Image): Image to get the controlnet conditioning images from.
        RETURNS:
            canny_image (Image): Canny edge detection image.
            openpose (Image): Openpose pose estimation image.
            softedge (Image): Soft edge detection image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Apply Canny edge detection to the image
        canny_image = calculate_canny(image)
        # Apply PIDiNet to the image
        softedge = calculate_softedge(image)
        # Apply Openpose pose estimation to the image
        openpose = Image.fromarray(OpenposeDetector(np.array(image)))
        return canny_image, openpose, softedge

    def resize_image(self, input_img: Image):
        """
        Resize an image to fit within max_width and max_height while preserving aspect ratio.

        Args:
            input_img (Image): Image to resize.
        Returns:
            Image: Resized image. According to self.generated_width and self.generated_height.

        """
        img = input_img
        max_width = self.generated_width
        max_height = self.generated_height
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height

        # Determine the new dimensions
        if original_width > max_width or original_height > max_height:
            # Resize to fit within max_width and max_height
            if aspect_ratio > 1:
                # Width is greater than height
                new_width = min(max_width, original_width)
                new_height = int(new_width / aspect_ratio)
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
            else:
                # Height is greater than or equal to width
                new_height = min(max_height, original_height)
                new_width = int(new_height * aspect_ratio)
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
        else:
            # Scale up to ensure larger dimension is at least 768 pixels
            if aspect_ratio > 1:
                # Width is greater than height
                new_width = max(max_width, original_width)
                new_height = int(new_width / aspect_ratio)
            elif aspect_ratio == 1:
                #Image is square
                new_width = max(max_width, original_width)
                new_height = new_width
            else:
                # Height is greater than or equal to width
                new_height = max(max_height, original_height)
                new_width = int(new_height * aspect_ratio)

        # Resize the image
        img = img.resize((new_width, new_height), Image.LANCZOS)
        return img

    def remove_background(self, image: Image):
        """
        Remove the background from an image using YOLO instance segmentation.
        Args:
            image (Image): Image to remove the background from.
        Returns:
            Image: Mask of the person in the image.
            Image: Image with the background removed.
        """
        person_mask = yolo_instance_segmentation(image)
        if isinstance(person_mask, np.ndarray):
            person_mask = Image.fromarray(person_mask)
        # Apply a Gaussian blur to the mask
        person_mask = person_mask.filter(ImageFilter.GaussianBlur(1))
        self.nobg_image_mask = person_mask
        # Apply the mask to the image to turn background transparent
        self.nobg_image = self.apply_transparency_mask(image, person_mask)
        return self.nobg_image_mask, self.nobg_image

    def overlay_background_foreground(self, background: Image, foreground: Image, mask: Image):
        """
        Overlay the foreground image on the background image using the mask.
        Args:
            background (Image): Background image.
            foreground (Image): Foreground image.
            mask (Image): Mask to apply to the foreground image.
        Returns:
            Image: Composite image with the foreground image overlayed on the background image.
        """
        foreground = Image.composite(foreground, background, mask)
        return foreground

    def apply_transparency_mask(self, image: Image, mask: Image):
        """
        Apply a transparency mask to an image. Make the background of the original image transparent.
        Args:
            image (Image): Image to apply the transparency mask to.
            mask (Image): Transparency mask to apply to the image.
        Returns:
            Image: Image with the transparency mask applied.
        """
        transparent_image = Image.new("RGBA", image.size)

        # Copy the RGB values from the original image
        transparent_image.paste(image, (0, 0))

        # Apply the mask to the alpha channel
        alpha = mask.point(lambda p: 255 if p > 0 else 0)  # Convert mask to alpha (255 for white, 0 for black)
        transparent_image.putalpha(alpha)
        return transparent_image

    def create_blank_canvas(self, color=(0, 0, 0, 255)):
        """
        Create a blank image canvas for moving the image around and resizing it.
        Args:
            color (tuple): Color of the canvas. (default is black)
        Returns:
            Image: Blank image canvas of size self.generated_width x self.generated_height.
        """
        # Create a blank image canvas using PIL
        image = Image.new("RGBA", (self.generated_width, self.generated_height), color=color)
        return image

    def move_and_resize_image_on_canvas(self, image, mask, move_x, move_y, resize_scale):
        """
        Move and resize the image on the canvas.
        Args:
            image (Image): Image to move and resize.
            mask (Image): Mask of the image.
            move_x (int): Number of pixels to move the image in the x-direction.
            move_y (int): Number of pixels to move the image in the y-direction.
            resize_scale (float): Scale factor to resize the image.
        Returns:
            Image: Image moved and resized on the canvas.
        """

        # Calculate the new size based on the resize scale
        new_width = int(image.width * resize_scale)
        new_height = int(image.height * resize_scale)
        # Resize the image
        resized_image = image.resize((new_width, new_height))
        resized_mask = mask.resize((new_width, new_height))

        # Calculate the new coordinates based on the movement
        new_x = move_x + (self.generated_width - resized_image.width) // 2
        new_y = move_y + (self.generated_height - resized_image.height) // 2

        if self.use_erode:
            self.image_canvas_eroded = self.create_blank_canvas(color=(0, 0, 0, 0))
            self.mask_canvas_eroded = self.create_blank_canvas(color=(0, 0, 0, 255))
            #Get the eroded image and eroded_mask. Discard eroded mask as not needed
            image_canvas_eroded = Image.fromarray(self.erode_boundaries(resized_image))
            mask_canvas_eroded = Image.fromarray(self.erode_boundaries(resized_mask))
            self.image_canvas_eroded.paste(image_canvas_eroded, (new_x, new_y))
            self.mask_canvas_eroded.paste(mask_canvas_eroded, (new_x, new_y))
            # Return the canvas image
            return self.image_canvas_eroded, self.mask_canvas_eroded

        else:
            # Create the blank canvas
            self.image_canvas = self.create_blank_canvas(color=(0, 0, 0, 0))
            self.mask_canvas = self.create_blank_canvas(color=(0, 0, 0, 255))
            # Paste the resized image onto the canvas at the new coordinates
            self.image_canvas.paste(resized_image, (new_x, new_y), mask=resized_image.split()[3])
            self.mask_canvas.paste(resized_mask, (new_x, new_y))
            # Return the canvas image
            return self.image_canvas, self.mask_canvas

    def erode_boundaries(self, image: Image, threshold_value: int = 205, erosion_size: int = 6):

        """
        Erode the boundaries of the image to remove the hard edges.
        Args:
            image (Image): Image to erode the boundaries of.
            threshold_value (int): Threshold value for the binary mask. (default is 205)
            erosion_size (int): Size of the erosion kernel. (default is 6)
        Returns:
            Image: Image with the boundaries eroded.
        """
        threshold_value = threshold_value
        erosion_size = erosion_size
        image = np.array(image)

        if image is None:
            raise ValueError("Image not found. Check the provided path.")

        if len(image.shape) == 2:
            img = Image.fromarray(image)
            img = img.convert("RGBA")
            image = np.array(img)

            # Ensure the image has an alpha channel
        if image.shape[2] != 4:
            image.convert("RGBA")

            # raise ValueError("Image does not have an alpha channel.")

            # Extract the alpha channel
        alpha_channel = image[:, :, 3]

        # Apply thresholding to the alpha channel to create a binary mask
        _, binary_mask = cv2.threshold(alpha_channel, threshold_value, 255, cv2.THRESH_BINARY)

        # Create a kernel for erosion
        kernel = np.ones((erosion_size, erosion_size), np.uint8)

        # Perform erosion on the binary mask
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

        # Update the alpha channel with the eroded mask
        image[:, :, 3] = eroded_mask

        # Save the resulting image
        return image

    def get_face_lora(self, prompt):
        """
        Get the face image from the LORA model.
        Args:
            positive_prompt (str): Positive prompt to generate the face image.
        Returns:
            str: Lora ID of the face image.
        """

        db = TinyDB(os.getenv('TINYDB_PATH'))
        facet_table = db.table(os.getenv('TINYDB_FACETABLE'))
        # Fetch all lora_ids from the table
        records = facet_table.all()
        # Check if any lora_id is present in the prompt
        for record in records:
            if record['lora_id'] in prompt:
                # Fetch the lora token and path
                print(record)
                return FaceLoraOptions(**record)

        return None

    def get_bg_id(self, prompt):
        """
        Get the background template from the positive prompt.
        Args:
            positive_prompt (str): Positive prompt to generate the background image.
        Returns:
            str: Lora ID of the background image.
        """

        db = TinyDB(os.getenv('TINYDB_PATH'))
        facet_table = db.table(os.getenv('TINYDB_BGTABLE'))
        # Fetch all lora_ids from the table
        records = facet_table.all()
        # Check if any lora_id is present in the prompt
        for record in records:
            if record['bg_id'] in prompt:
                # Fetch the lora token and path
                print(record)
                return record['bg_id']

        return None

    def load_lora(self, pipeline, lora_path):
        """
        Load the LORA weights into the pipeline.
        Args:
            pipeline (AdInpaintCnPipeline): Pipeline to load the LORA weights into.
            lora_path (str): Path to the LORA weights.
        Returns:
            None
        """
        pipeline.load_lora_weights(lora_path)

    def load_textual_inversions(self):
        """
        Load the textual inversions into the pipeline.
        """
        try:
            text_inversion_path_list = glob.glob(os.path.join(self.config['Textual_Inversion_Path'], '*.pt'))
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
            logging.error(f"Error while loading textual inversion: {e}")

    def remove_lora(self, pipeline):
        """
        Remove the LORA weights from the pipeline.
        Args:
            pipeline (AdInpaintCnPipeline): Pipeline to remove the LORA weights from.
        """
        pipeline.unload_lora_weights()

    def restore_area(self, original_mask: Image, genrated_image: Image):
        """
        Restore the area of the original image that was removed by the mask
        Args:
            original_mask (Image): Mask of the original image.
            genrated_image (Image): Generated image.
        Returns:
            Image: Image with the original area restored.
        """

        return

    def __call__(self,
                 original_img: Image,
                 original_mask: Image,
                 controlnet_images: list[Image.Image],
                 generation_settings: json,
                 bg_lora_path: str,
                 ) -> tuple[Any, Any]:
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

        self.load_lora(self.checkpoint, bg_lora_path)
        # load textual inversion
        self.load_textual_inversions()

        # fix the seed value
        generator = torch.Generator(device='cuda')
        generator.manual_seed(generation_settings['seed'])

        #get face lora
        if generation_settings['face_generate']:
            face_lora = self.get_face_lora(generation_settings['inpaint_positive_prompt'])

        else:
            face_lora = self.get_face_lora('model_a')

        if face_lora is not None:
            self.load_lora(self.checkpoint.inpaint_pipeline, face_lora.lora_path)

        # Add token to the positive prompt
        # generation_settings['positive_prompt'] = face_lora.lora_token + generation_settings['positive_prompt']

        #convert prompts to embeddings
        prompt_embeds = self.compel(generation_settings['positive_prompt'])
        negative_prompt_embeds = self.compel(generation_settings['negative_prompt'])

        #Inpainting prompts to embeddings
        inpaint_prompt_embeds = self.compel(face_lora.lora_token)

        common = {
            "negative_prompt_embeds": negative_prompt_embeds,
            "num_inference_steps": generation_settings['num_inference_steps'],
            "width": self.generated_width,
            "height": self.generated_height,
            "generator": generator,
            "clip_skip": generation_settings['clip_skip'],
            "num_images_per_prompt": 1,
        }

        enhanced_images = self.checkpoint(common=common,
                                          txt2img_only={"prompt_embeds": prompt_embeds,
                                                        "image": original_img,
                                                        "mask_image": original_mask,
                                                        "strength": 1.0,
                                                        "guidance_scale": generation_settings['guidance_scale'],
                                                        "guess_mode": True,
                                                        "control_image": controlnet_images,
                                                        "controlnet_conditioning_scale": generation_settings[
                                                            'controlnet_conditioning_scale'],
                                                        "cross_attention_kwargs": {
                                                            "scale": float(0.9)
                                                        }
                                                        },
                                          inpaint_only={"prompt_embeds": inpaint_prompt_embeds,
                                                        "strength": 0.6,
                                                        "image": original_img,
                                                        "guidance_scale": 5,
                                                        "cross_attention_kwargs": {
                                                            "scale": float(0.9)
                                                        }
                                                        },
                                          detectors=[yolo_detector])

        self.remove_lora(self.checkpoint)
        self.remove_lora(self.checkpoint.inpaint_pipeline)

        return enhanced_images.images, enhanced_images.init_images


# if __name__ == "__main__":
#     img = Image.open("/home/hamna/Database/users/hamnatest/images/25_DBJrC woman/003.JPEG")
#     bg = Image.open("/home/hamna/Downloads/001.png")
#     fg = FormsGenerator()
#     resized_img = fg.resize_image(img)
#     # resized_bg = fg.resize_image(bg)
#     mask, img = fg.remove_background(resized_img)
#     mask.save('/home/hamna/testmask.png')
#     img.save('/home/hamna/testimg.png')
#     img_c, mask_c = fg.move_and_resize_image_on_canvas(img, mask, 0, 200, 1)
#     img_c.save('/home/hamna/testerode.png')
#     mask_c.save('/home/hamna/testmaskerode.png')
#
#     overlayed = fg.overlay_background_foreground(bg, img_c, img_c)
#     overlayed.save('/home/hamna/testoverlay.png')
#
#     x, y = fg.get_controlnet_imgs(overlayed)
#     x.save('/home/hamna/testcanny.png')
#     y.save('/home/hamna/testopenpose.png')
#     # x = fg.get_face_lora("Best quality, masterpiece, ultra high res, (photorealistic:1.4), 1gir, beautiful asian
#     # woman, standing in a cafe , ((realistic skin)), high quality, studio lighting, photoshoot, uniform light, gyj,
#     # gyj woman")
#     print("dd")
