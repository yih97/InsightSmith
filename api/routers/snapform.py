
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import requests
import fastapi.responses
import logging
from modules.snapform.snapform_instances import (SnapformGenerateOptions, SnapformGenerateResponse)
from modules.snapform.snapform import run_snapform
from dependency_injector.wiring import inject, Provide
from modules.services.s3_management import S3Service
from typing import Annotated
router = APIRouter(
    prefix='/snapform',
    tags=["snapform, generate, v1.0.0"]
)


def add_task(options: SnapformGenerateOptions, s3_service: S3Service):
    try:
        response = run_snapform(options, s3_service)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def get_s3_service(bucket_name: str):
    return S3Service(bucket_name=bucket_name)

@router.get('/generate')
@inject
def snapform(options: SnapformGenerateOptions, background_task: BackgroundTasks,
             s3_service: Annotated[S3Service, Depends(lambda: get_s3_service("snapform"))]):
    """
        This endpoint is used to generate images using the given options for forms generator.
    """
    try:
        #Return response while running in the background
        background_task.add_task(add_task, options, s3_service)
        return {"message": "Request accepted. The process is running in the background."}

        #To be used when running as one process no return requested.
        # response = run_snapform(options)
        # if response.status_code == 200:
        #     data = response.json()
        #     return data
        # else:
        #     raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))