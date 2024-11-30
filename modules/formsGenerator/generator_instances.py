from pydantic import BaseModel, Field, create_model

from typing import Dict, List
from typing import Dict, List, Optional

class GeneratorResponse(BaseModel):
    generated_images: List[str] = Field(default=None, title="Generated Image", description="The generated image in base64 format.")
    enhanced_images: List[str] = Field(default=None, title="Enhanced Image", description="The enhanced image in base64 format.")

class GenerateOptions(BaseModel):
    image: str
    mask: str
    positive_prompt: str
    negative_prompt: str
    use_template: bool
    bg_id: str = "roof"
    batch: int
    size: int
    denoising_strength: float
    face_generate: bool

class FaceLoraOptions(BaseModel):
    lora_id: str
    lora_token: str
    lora_path: str