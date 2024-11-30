from pydantic import BaseModel


class SnapformGenerateOptions(BaseModel):
    gender: str
    user_uuid: str
    theme: str
    trainingImgs_uuid:str
    photocard_uuid:str


class SnapformGenerateResponse(BaseModel):
    status: str
    message: str
    url: str

class SnapformModel(BaseModel):
    model_id: str
    model_category: str
    model_path: str
    model_token: str
    model_repo: str