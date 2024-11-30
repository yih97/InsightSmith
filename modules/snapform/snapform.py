import os
import shutil

# from modules.s3 import *
import logging
from internal.pipelines.snapform import SnapForm
# from modules.s3 import S3Service
from modules.services.s3_management import S3Service
from modules.snapform.snapform_instances import SnapformGenerateOptions
from dotenv import load_dotenv
from internal.training_scripts.lora_json import train_settings, load_json
import json
from internal.components.utils import load_and_check_json_files
import random
import string
from modules.service.tinydb_mangement import data_form, insert_data, exist_lora
from internal.training_scripts.wd14_caption import caption_images
from tinydb import TinyDB, Query
load_dotenv()

db = TinyDB(os.getenv('TINYDB_PATH'))
# DB- table name : snapform
table=db.table("snapform")

snapform = SnapForm(adetailer=True)
def check_lora_exists(uuid: str):
    try:
        logging.info('Checking if user lora exists')
        lora_path = os.path.join(os.environ.get('LORA_PATH'),
                                 uuid + ".safetensors")
        exists = os.path.exists(lora_path)

        if exists:
            return exists, lora_path
        else:
            lora_path = exist_lora(uuid)
            return exists, lora_path

    except Exception as e:
        logging.error(f"Error checking if user lora exists: {e}")
        raise e


def run_snapform(options: SnapformGenerateOptions, s3_service: S3Service):
    try:
        logging.info("Running snapform Generator")
        #check if the user lora exists
        exists, lora_path = check_lora_exists(options.user_uuid)

        if exists:
            logging.info('User lora exists. Generating images')

        else:
            logging.info('User lora does not exist. Training new model')
            trained_lora_path, token = train_model(options, s3_service)

            #To-do: Check if training complete before copying an d moving forward
            shutil.copy(trained_lora_path, lora_path)

            # Update the user's lora path in the database
            data = data_form(model_category= "FACE",model_id=options.user_uuid, token=token)
            insert_data(data)

            # db.insert({'user_uuid': options.user_uuid, 'lora_path': lora_path, 'token': token})
            logging.info('Model trained and saved successfully')

        generate_images(options, s3_service)

    except Exception as e:
        logging.error(f"Error running snapform: {e}")
        raise e


def train_model(options: SnapformGenerateOptions, s3_service: S3Service):
    # Set the local and s3 paths for the new user
    local_path = os.path.join(os.environ.get('LOCAL_DATA_PATH'), options.user_uuid)
    img_dir = os.path.join(local_path, 'images')
    model_dir = os.path.join(local_path, 'models')
    log_dir = os.path.join(local_path, 'logs')

    # create directories
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    s3_path = os.path.join(os.environ.get('S3_Path_Users'), options.user_uuid,
                           options.trainingImgs_uuid)

    # Load the train_lora settings.json
    lora_setting_path = os.environ.get('TRAIN_SETTINGS_PATH')
    settings = load_json(lora_setting_path)

    # Replace the user path in the settings
    settings['logging_dir'] = log_dir
    settings['output_dir'] = model_dir
    settings['train_data_dir'] = img_dir
    settings['output_name'] = options.user_uuid

    # Save the settings to the user's local path for backup
    with open(os.path.join(local_path, 'settings.json'), 'w') as f:
        json.dump(settings, f)

    # Creating random phrase as token
    token = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    # Create directory with steps and token
    if options.gender.lower() == 'f':
        img_dir = os.path.join(img_dir, f"{os.environ.get('STEPS_F')}_{token} woman")
    else:
        img_dir = os.path.join(img_dir, f"{os.environ.get('STEPS_M')}_{token} man")

    os.makedirs(img_dir, exist_ok=True)
    # Download the user's data from s3
    s3_service.download_images(s3_path, img_dir)

    # Train the model
    # Todo: Implement Auto-captioning when training
    logging.info('Auto caption used wd14')
    tag_cmd = caption_images(train_data_dir=img_dir, prefix_tags=[token], force_download=False, onnx=True)
    logging.info('Training the model')
    train_cmd = train_settings(**settings)

    from internal.training_scripts.library.class_command_executor import CommandExecutor
    cmd_executor = CommandExecutor()
    for cmd in [tag_cmd, train_cmd]:
        cmd_executor.execute_command(cmd)

    logging.info('Model trained successfully')
    final_lora_path = select_epoch(options.gender, model_dir)
    return final_lora_path, token


def generate_images(options: SnapformGenerateOptions, s3_service: S3Service):
    global snapform
    try:
        # Do generation
        logging.info('Generating images')

        generation_id = options.photocard_uuid  #snapform.generate_id()
        templates = load_and_check_json_files(os.path.join(snapform.config['THEME_PATH'], options.theme))
        for template in templates:
            logging.info(f'Generating Image for {template}')
            with open(template) as json_file:
                settings = json.load(json_file)
            try:
                #Add values to local backend database
                User = Query()
                user = db.search(User.user_uuid == options.user_uuid)[0]
                # if 조건문 추가(점수 만족하지 못하면 재생성)
                enhanced, generated, inpaint = snapform(options.user_uuid, options.gender,user['token'],settings)
                # enhanced img 점수 계산
                logging.info(f'Images generated: {enhanced}')
            except IndexError as e:
                logging.error(f"Error generating images: {e}")
                raise IndexError(f"Error generating images: {e}")

            logging.info('Uploading images to s3')

            snapform.save_images(generated, enhanced)  #temporary

            s3_service.upload_generated_images(f"{os.environ.get('S3_Path_Gen')}/{options.user_uuid}/{generation_id}",
                                               enhanced)
            logging.info(
                f"Image generated and uploaded to s3: {os.environ.get('S3_Path_Gen')}/{options.user_uuid}/{generation_id}")

        s3_service.upload_to_sqs(options.user_uuid, generation_id)
        logging.info(f"Uploaded to SQS with id: {generation_id}")


    except Exception as e:
        logging.error(f"Error generating images: {e}")
        raise e


def select_epoch(gender: str, model_dir: str) -> str | None:
    """
    Select the best epoch number for the model
    Args:
        gender :
        model_dir :

    Returns:
        selected_epoch : str
    """
    logging.info('Selecting best epoch number')

    if gender.lower() == 'f':
        epoch_number = os.environ.get('EPOCH_NUMBER_F') + '.safetensors'
    else:
        epoch_number = os.environ.get('EPOCH_NUMBER_M') + '.safetensors'

    all_epochs = os.listdir(model_dir)
    matching_files = [os.path.join(model_dir, file) for file in all_epochs if str(epoch_number) in file]

    selected_epoch = matching_files[0] if matching_files else None

    if selected_epoch is None:
        logging.error('No epoch number found')
        raise Exception('No epoch number found')
    else:
        logging.info(f'Epoch number selected: {selected_epoch}')
        return selected_epoch

