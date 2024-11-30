##Router for Segment Anything used by forms Generator

import numpy as np
from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, Depends, HTTPException
import requests
from typing import List, Annotated
from internal.components.samexporter.samexporter.sam_onnx import SegmentAnythingONNX
from PIL import Image
from pydantic import BaseModel
import logging
from modules.formsGenerator.s3_service import S3Service
import os
import torch
from dotenv import load_dotenv
# from config import ROOT_DIR

load_dotenv()
router = APIRouter(
    prefix='/SAM',
    tags=["SAM"]
)

model_zoo = os.getenv('MODEL_ZOO')
encoder_model = os.path.join(model_zoo,"sam/sam_vit_h_4b8939.encoder.onnx")
decoder_model = os.path.join(model_zoo,"sam/sam_vit_h_4b8939.decoder.onnx")

# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     level=logging.INFO  # Set the logging level to INFO or desired level
# )

logging.info('loading model')

# model = SegmentAnythingONNX(
#     encoder_model,
#     decoder_model,
# )

logging.info('loading model completed')

class embedding_obj(BaseModel):
    image_embedding: List

@router.get('/embedding')
@inject
def get_bg_template(img_url: str, s3_service: Annotated[S3Service, Depends(S3Service)]):
    embeddings = None
    try:
        try:
            logging.info('Fetching Image')
            image = Image.open(requests.get(img_url, stream=True).raw)

            if not image.mode == 'RGB':
                image = image.convert('RGB')

            logging.info('Fetching Image complete')

        except Exception as e:
            logging.error("Exception occurred:", e)
            return "Failed to download Input image"

        logging.info('Calculating embedding')
        img = np.array(image)
        embeddings = model.encode(img)
        embedding = embeddings['image_embedding'].tolist()
        shape = embeddings['image_embedding'].shape
        flattened = embeddings['image_embedding'].flatten().tolist()
        logging.info('Calculating embedding complete')
        response_load = {
            # "image_embedding": embedding,
            "shape": shape,
            "flattened": flattened
        }
        # response = embedding_obj(**response_load)
        url = s3_service.upload_file("stable_diffusion/embedding/", response_load)
        logging.info('Return response embedding')
        torch.cuda.empty_cache()

        return url

    except Exception as e:
        logging.error("Exception occurred:", e)
        return "Calculating Embedding failed"