import os
import json
import logging
import uuid
from datetime import datetime

from modules.formsGenerator.generator_instances import GenerateOptions
from modules.formsGenerator.s3_service import S3Service, fetch_bg_template, fetch_image
from internal.pipelines.formsGenerator import FormsGenerator
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

# fg = FormsGenerator()


def get_lora_path(lora_name):
    lora_path = os.path.join(os.environ.get('generator_Lora_Paths'), lora_name + ".safetensors")
    return lora_path


def run_forms_generator(options: GenerateOptions, s3_service: S3Service):
    try:
        logging.info("Running forms Generator")
        logging.info('I am here')
        logging.info(f"Requested generation with: {options.positive_prompt, options.face_generate}")

        ###What to do if use_template is false backend hack
        if options.use_template is False:
            options.bg_id = fg.get_bg_id(options.positive_prompt)
        #Check if template already downloaded
        if not os.path.exists(os.path.join(os.getenv('generator_bg_template'), options.bg_id + '.png')):
            logging.info('Downloading BG Template')
            s3_service.download_bg_template(options.bg_id, os.getenv('s3_bg_template'))
        else:
            logging.info('BG Template already downloaded')

        #Fetch BG Templates
        template_path = os.path.join(os.getenv('generator_bg_template'), options.bg_id + '.png')
        template_json_path = os.path.join(os.getenv('generator_bg_template'), options.bg_id + '.json')

        bg_template = Image.open(template_path)
        bg_properties = json.load(open(template_json_path))

        # bg_template, bg_properties = s3_service.download_bg_template(options.bg_id, os.getenv('s3_bg_template'))
        # bg_lora_name = bg_properties['bg_lora_name']
        bg_lora_path = get_lora_path(options.bg_id)

        #Preprocess Image
        original_img = fetch_image(options.image)
        resized_img = fg.resize_image(original_img)
        original_mask = fetch_image(options.mask)
        original_mask_resized = fg.resize_image(original_mask)
        mask, img = fg.remove_background(resized_img)
        img_c, mask_c = fg.move_and_resize_image_on_canvas(img, mask, bg_properties['move_x'], bg_properties['move_y'],
                                                           bg_properties['resize'])
        overlayed = fg.overlay_background_foreground(bg_template, img_c, img_c)
        canny, openpose, softedge = fg.get_controlnet_imgs(overlayed)

        #Generation_settings
        bg_properties['inpaint_positive_prompt'] = options.positive_prompt
        # bg_properties['negative_prompt'] = options['negative_prompt']
        bg_properties['face_generate'] = options.face_generate

        #Generate Images
        mask_c = mask_c.convert('L')
        mask_c = ImageOps.invert(mask_c.convert('L')).convert('1')
        img_c = img_c.convert('RGB')

        enhanced, generated = fg(img_c,mask_c,[openpose,softedge, canny],
                                 bg_properties,
                                 bg_lora_path)

        #Upload Images
        # id = datetime.now().strftime("%Y%m%d%H%M%S")
        # s3_service.upload_image(generated[0], f'generator/{id}_generated.png')
        # generated[0].save(f'/home/ha mna/Database/results/generator/{id}_generated.png')
        # enhanced[0].save(f'/home/hamna/Database/results/generator/{id}_enhanced.png')

        if options.face_generate is False:
            enhanced = generated

        return generated, enhanced


    except Exception as e:
        logging.error(f"Error running forms generator: {e}")
        raise e


if __name__ == '__main__':
    options = {
        "image": "https://dfmarqni2tgyh.cloudfront.net/stable_diffusion/input/003.png",
        "mask": "https://dfmarqni2tgyh.cloudfront.net/stable_diffusion/input/003-mask_full.png",
        "positive_prompt": "Best quality, masterpiece, ultra high res, studio lighting, photoshoot, uniform light, suji",
        "negative_prompt": "bad hands, dark light, disfigured hands",
        "use_template": "True",
        "bg_id": "street",
        "batch": 1,
        "size": 1024,
        "denoising_strength": 1.0,
        "face_generate": "True"
    }

    run_forms_generator(GenerateOptions(**options), S3Service())
