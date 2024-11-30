
import io
import json

import boto3
from pathlib import Path
from PIL import Image
import os
from dotenv import load_dotenv
import requests

# load_dotenv()


AWS_S3_REGION = os.getenv('GENERATOR_AWS_S3_REGION')
AWS_S3_ACCESS_KEY_ID= os.getenv('GENERATOR_AWS_S3_ACCESS_KEY_ID')
AWS_S3_SECRET_ACCESS_KEY= os.getenv('GENERATOR_AWS_S3_SECRET_ACCESS_KEY')
AWS_S3_PUBLIC_BUCKET_NAME= os.getenv('GENERATOR_AWS_S3_PUBLIC_BUCKET_NAME')
AWS_CLOUDFRONT_URL= os.getenv('GENERATOR_AWS_CLOUDFRONT_URL')

def fetch_image(url) -> Image.Image:
    image = Image.open(requests.get(url, stream=True).raw)
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    return image
def fetch_bg_template(id):
    # download image from datasource
    img_path = "stable_diffusion/bg_template/" + id + ".png"
    img_url = "https://dfmarqni2tgyh.cloudfront.net/" + img_path

    # download template settings fron datasource
    options_path = "stable_diffusion/bg_template/" + id + ".json"
    options_url = "https://dfmarqni2tgyh.cloudfront.net/" + options_path
    bg_template = fetch_image(img_url)
    response = requests.get(options_url)
    options = response.json()
    return bg_template, options


class S3Service:
    def __init__(self):
        self.bucket = boto3.resource(
            's3',
            aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY,
            region_name=AWS_S3_REGION
        ).Bucket(
            AWS_S3_PUBLIC_BUCKET_NAME
        )

    def get_filename_json(self, path):
        # path = stable_diffusion/bg_change/ or stable_diffusion/face_change/ or stable_diffusion/clothes_change/
        count = len(list(self.bucket.objects.filter(Prefix=path)))
        count = count + 1
        return path + str(count) + '.json'

    def upload_file(self, path: str, data):

        json_obj = json.dumps(data)
        suffix = Path(path).suffix
        extension = suffix.replace('.', '')
        key = self.get_filename_json(path)
        self.bucket.put_object(
            Key=key,
            Body=json_obj
        )

        return AWS_CLOUDFRONT_URL + key


    def upload_image(self, path: str, image: Image) -> str:

        #convert Image to buffer string
        img_stream = io.BytesIO()
        image.save(img_stream, format='PNG')
        buffer = img_stream.getvalue()

        #upload image buffer to S3
        suffix = Path(path).suffix
        extension = suffix.replace('.', '')
        key = self.get_filename(path)

        self.bucket.put_object(
            Key=key,
            Body=buffer,
            ContentType='image/' + extension
        )
        return AWS_CLOUDFRONT_URL + key


    def get_filename(self, path):
        #path = stable_diffusion/bg_change/ or stable_diffusion/face_change/ or stable_diffusion/clothes_change/
        count = len(list(self.bucket.objects.filter(Prefix=path)))
        count = count + 1
        return path+str(count)+'.png'


    def download_bg_template(self, bg_id, base_path):
        png_s3_path = os.path.join(base_path, f"{bg_id}.png")
        json_s3_path = os.path.join(base_path, f"{bg_id}.json")

        # Construct the full local paths for .png and .json files
        local_dir = os.path.join(os.getenv('generator_bg_template'))
        os.makedirs(local_dir, exist_ok=True)

        png_local_path = os.path.join(local_dir, f"{bg_id}.png")
        json_local_path = os.path.join(local_dir, f"{bg_id}.json")

        try:
            # Download the .png file
            self.bucket.download_file(png_s3_path, png_local_path)
            print(f"Downloaded {bg_id}.png to {png_local_path}")

            # Download the .json file
            self.bucket.download_file(json_s3_path, json_local_path)
            print(f"Downloaded {bg_id}.json to {json_local_path}")

        except Exception as e:
            print(f"Error downloading files: {e}")




    def get_bg_templates(self):
        path = "stable_diffusion/bg_template/"
        backgrounds = self.bucket.objects.filter(Prefix=path)
        bg_name = []
        for bg in backgrounds:
            if bg.key.endswith('.json'):
                bg_name.append(bg.key)

        return bg_name

