import json
import mimetypes
import uuid

import boto3
import io
from dotenv import load_dotenv
import logging
import os
from PIL import Image


class S3Service:
    """Class to manage S3 operations for multiple buckets"""

    def __init__(self, bucket_name='snapform'):
        """Initialize S3 service with bucket name
        Args:
            bucket_name (str): Name of the bucket to be used. default is Snapform
        """
        load_dotenv()
        self.AWS_S3_REGION = os.getenv('AWS_S3_REGION')
        self.AWS_S3_ACCESS_KEY_ID = os.getenv('AWS_S3_ACCESS_KEY_ID')
        self.AWS_S3_SECRET_ACCESS_KEY = os.getenv('AWS_S3_SECRET_ACCESS_KEY')
        try:
            if bucket_name == 'snapform':
                self.AWS_S3_PUBLIC_BUCKET_NAME = os.getenv('AWS_S3_PUBLIC_BUCKET_NAME_SNAPFORM')
                self.AWS_CLOUDFRONT_URL = os.getenv('AWS_CLOUDFRONT_URL_SNAPFORM')
            elif bucket_name == 'generator':
                self.AWS_S3_PUBLIC_BUCKET_NAME = os.getenv('AWS_S3_PUBLIC_BUCKET_NAME_GENERATOR')
                self.AWS_CLOUDFRONT_URL = os.getenv('AWS_CLOUDFRONT_URL_GENERATOR')
            else:
                logging.error("Invalid bucket name")
                raise ValueError("Invalid bucket name")
        except Exception as e:
            logging.error(f"Error in S3Service init: {e}")
            raise e
        try:
            self.bucket = boto3.resource(
                's3',
                aws_access_key_id=self.AWS_S3_ACCESS_KEY_ID,
                aws_secret_access_key=self.AWS_S3_SECRET_ACCESS_KEY,
                region_name=self.AWS_S3_REGION
            ).Bucket(
                self.AWS_S3_PUBLIC_BUCKET_NAME
            )
        except Exception as e:
            logging.error(f"Error in bucket init: {e}")
            raise e

    def upload_generated_images(self, path: str, images: list) -> str:
        """
        Upload multiple images to S3 bucket
        :param path:
        :param images:
        :return:
        """
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
        return self.AWS_CLOUDFRONT_URL + path

    def upload_image(self, path: str, image: Image) -> str:
        """Upload an image to S3 bucket

        Args:
            path (str): Path where the image will be uploaded in the S3 bucket.
            image (Image): PIL Image object to be uploaded.

        Returns:
            str: URL of the uploaded image.
        """
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            #give a random name to the image
            img_name = uuid.uuid4()
            key = os.path.join(path, str(img_name) + '.png')
            self.bucket.put_object(Key=key, Body=img_byte_arr, ContentType='image/png')
            image_url = f"{self.AWS_CLOUDFRONT_URL}/{path}"
            logging.info(f"Image uploaded successfully: {image_url}")
            return image_url
        except Exception as e:
            logging.error(f"Error uploading image: {e}")
            raise e

    def download_images(self, s3_directory_path: str, local_directory: str):
        """Download all images from a directory in S3 bucket to a local directory
        Args:
            s3_directory_path (str): Path of the directory in the S3 bucket.
            local_directory (str): Path of the local directory where images will be downloaded.

        """
        try:
            #To-do: Check if user images exist in the S3 directory
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

        except Exception as e:
            logging.error(f"Error downloading images: {e}")
            raise e

    def download_image(self, path: str) -> Image:
        """Download an image from S3 bucket

        Args:
            path (str): Path of the image in the S3 bucket.

        Returns:
            Image: PIL Image object of the downloaded image.
        """
        try:
            obj = self.bucket.Object(path).get()
            img_byte_arr = io.BytesIO(obj['Body'].read())
            image = Image.open(img_byte_arr)
            logging.info(f"Image downloaded successfully from {path}")
            return image
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            raise e

    def upload_file(self, path: str, data: dict) -> str:
        """Upload a JSON file to S3 bucket

        Args:
            path (str): Path where the JSON file will be uploaded in the S3 bucket.
            data (dict): Dictionary to be uploaded as JSON.

        Returns:
            str: URL of the uploaded JSON file.
        """
        try:
            json_data = json.dumps(data)
            self.bucket.put_object(Key=path, Body=json_data, ContentType='application/json')
            file_url = f"{self.AWS_CLOUDFRONT_URL}/{path}"
            logging.info(f"JSON file uploaded successfully: {file_url}")
            return file_url
        except Exception as e:
            logging.error(f"Error uploading JSON file: {e}")
            raise e

    def download_file(self, path: str) -> dict:
        """Download a JSON file from S3 bucket

        Args:
            path (str): Path of the JSON file in the S3 bucket.

        Returns:
            dict: Dictionary of the downloaded JSON file.
        """
        try:
            obj = self.bucket.Object(path).get()
            json_data = obj['Body'].read().decode('utf-8')
            data = json.loads(json_data)
            logging.info(f"JSON file downloaded successfully from {path}")
            return data
        except Exception as e:
            logging.error(f"Error downloading JSON file: {e}")
            raise e

    def upload_to_sqs(self, received_uid, generation_id):
        # Initialize SQS client with your AWS credentials
        sqs = boto3.client('sqs',
                           aws_access_key_id=self.AWS_S3_ACCESS_KEY_ID,
                           aws_secret_access_key=self.AWS_S3_SECRET_ACCESS_KEY,
                           region_name=self.AWS_S3_REGION
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
