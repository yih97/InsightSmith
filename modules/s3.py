import io
import uuid

import boto3
from pathlib import Path
from PIL import Image
import json
from dotenv import load_dotenv
import os
import mimetypes
import logging

"""Load S3 details from .env file"""
load_dotenv()
AWS_S3_REGION = os.getenv('AWS_S3_REGION')
AWS_S3_ACCESS_KEY_ID = os.getenv('AWS_S3_ACCESS_KEY_ID')
AWS_S3_SECRET_ACCESS_KEY = os.getenv('AWS_S3_SECRET_ACCESS_KEY')
AWS_S3_PUBLIC_BUCKET_NAME = os.getenv('AWS_S3_PUBLIC_BUCKET_NAME')
AWS_CLOUDFRONT_URL = os.getenv('AWS_CLOUDFRONT_URL')


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

    def upload_image(self, path: str, image: Image) -> str:
        # convert Image to buffer string
        img_stream = io.BytesIO()
        image.save(img_stream, format='PNG')
        buffer = img_stream.getvalue()

        # upload image buffer to S3
        suffix = Path(path).suffix
        extension = suffix.replace('.', '')
        key = self.get_filename(path)

        self.bucket.put_object(
            Key=key,
            Body=buffer,
            ContentType='image/' + extension
        )
        return AWS_CLOUDFRONT_URL + key

    def upload_generated_images(self, path: str, images: list) -> str:
        # convert Image to buffer string
        for i,image in enumerate(images):
            img_name = uuid.uuid4()
            img_stream = io.BytesIO()
            image.save(img_stream, format='PNG')
            buffer = img_stream.getvalue()

            key = os.path.join(path,str(img_name) + '.png')

            self.bucket.put_object(
                Key=key,
                Body=buffer,
                ContentType='image/png'
            )
        return AWS_CLOUDFRONT_URL + path


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

    def get_filename(self, path):
        # path = stable_diffusion/bg_change/ or stable_diffusion/face_change/ or stable_diffusion/clothes_change/
        count = len(list(self.bucket.objects.filter(Prefix=path)))
        count = count + 1
        return path + str(count) + '.png'

    def download_images(self, s3_directory_path:str, local_directory:str):
        response = self.bucket.objects.filter(Prefix=s3_directory_path)
        # Iterate over each object in the directory
        for obj in response:
            # Get the key (filename) of the object
            key = obj.key

            # Extract the filename from the key
            filename = os.path.basename(key)

            # Check if the object is an image based on its MIME type
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type and mime_type.startswith('image'):
                # Download the image from S3 to the local directory
                local_filepath = os.path.join(local_directory, filename)
                if not os.path.exists(local_filepath):
                    os.makedirs(os.path.dirname(local_filepath), exist_ok=True)

                self.bucket.download_file(key, local_filepath)

                print(f"Downloaded {filename} to {local_filepath}")

    def upload_to_sqs(self,received_uid, generation_id):
        # Initialize SQS client with your AWS credentials
        sqs = boto3.client('sqs',
                           aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY,
                           region_name=AWS_S3_REGION
                           )

        # Define the message body in the required format
        message_body = {
            "uuid": str(received_uid),
            "photo_id": str(generation_id),
        }

        json_message = json.dumps(message_body)

        # Send the message to the specified SQS queue
        response = sqs.send_message(
            QueueUrl='snapform-result',
            MessageBody=json_message
        )

        logging.info('uploaded generation_id to SQS queue')

        return response


if __name__=="__main__":
    image = Image.open("/home/hamna/Database/Tem/city/city1/city1.png")
    s3 = S3Service()
    s3.upload_image("test/", image)