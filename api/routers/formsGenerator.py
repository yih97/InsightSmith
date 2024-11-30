from dependency_injector.wiring import inject, Provide

from typing import Annotated

import gc
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from modules.formsGenerator.generator_instances import GenerateOptions, GeneratorResponse
from modules.formsGenerator.formsGenerator import run_forms_generator
from modules.formsGenerator.s3_service import S3Service
import logging

router = APIRouter(
    prefix='/Generate',
    tags=["generate"]
)


@router.post('/change', summary='Swap face of given image(url)')
@inject
def generate_image(options: GenerateOptions, s3_service: Annotated[S3Service, Depends(S3Service)]):
    """
    This endpoint is used to generate images using the given options for forms generator.
    """

    try:

        generated_images, enhanced_images = run_forms_generator(options, s3_service)
        generated_list = []
        enhanced_list = []

        for image in generated_images:
            result_url = s3_service.upload_image(
                path="stable_diffusion/generate/",
                image=image
            )

            generated_list.append(result_url)

        for image in enhanced_images:
            result_url_ = s3_service.upload_image(
                path="stable_diffusion/enhanced/",
                image=image
            )

            enhanced_list.append(result_url_)

        response_load = {
            "generated_images": generated_list,
            "enhanced_images": enhanced_list
        }

        response = GeneratorResponse(**response_load)

        del enhanced_images
        del generated_images
        gc.collect()
        return response
    except Exception as err:
        raise HTTPException(detail=str(err), status_code=500)
