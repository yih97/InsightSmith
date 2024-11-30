import cv2
import numpy as np
from PIL import Image, ImageOps
import logging
import os
import safetensors
import torch
from urllib.parse import urlparse
from typing import Optional, Tuple
import math
import json

def convert_image_to_numpy(image: Image.Image) -> np.array:
    """
    Convert PIL image to numpy array
    :param image: PIL Image in RGB mode and size HxWx3
    :return np_image: numpy array with size HxWx3 in BGR mode
    """

    np_image = np.array(image)[:, :, ::-1]
    return np_image


def convert_numpy_to_image(np_image: np.array) -> Image.Image:
    """
    Convert numpy array to PIL image
    :param np_image: numpy array with size HxWx3
    :return image: PIL Image in RGB mode
    """

    img = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    image = Image.fromarray(img, "RGB")
    return image

unsafe_torch_load = torch.load
def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = unsafe_torch_load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    logging.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict

def get_state_dict(d):
    return d.get("state_dict", d)

def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file
def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a torch tensor
    :param image: PIL Image
    :return tensor: torch tensor
    """
    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)

def hwc2bchw(images: torch.Tensor) -> torch.Tensor:
    """
    Convert image from HWC to BCHW
    Args:
        images (torch.Tensor): input image batch.
    Returns:
        torch.Tensor: BCHW image batch.
    """
    return images.unsqueeze(0).permute(0, 3, 1, 2)


def bchw2hwc(images: torch.Tensor, nrows: Optional[int] = None, border: int = 2,
             background_value: float = 0) -> torch.Tensor:
    """ make a grid image from an image batch.

    Args:
        images (torch.Tensor): input image batch.
        nrows: rows of grid.
        border: border size in pixel.
        background_value: color value of background.
    """
    assert images.ndim == 4  # n x c x h x w
    images = images.permute(0, 2, 3, 1)  # n x h x w x c
    n, h, w, c = images.shape
    if nrows is None:
        nrows = max(int(math.sqrt(n)), 1)
    ncols = (n + nrows - 1) // nrows
    result = torch.full([(h + border) * nrows - border,
                         (w + border) * ncols - border, c], background_value,
                        device=images.device,
                        dtype=images.dtype)

    for i, single_image in enumerate(images):
        row = i // ncols
        col = i % ncols
        yy = (h + border) * row
        xx = (w + border) * col
        result[yy:(yy + h), xx:(xx + w), :] = single_image
    return result

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def load_and_check_json_files(directory):
    required_keys = ["positive_prompt", "negative_prompt", "controlnet_images",
                     "num_inference_steps", "guidance_scale", "clip_skip",
                     "num_images_per_prompt", "use_compel", "height", "width",
                     "controlnet_conditioning_scale", "seed"]

    json_files = find_json_files(directory)
    for file in json_files:
        with open(file) as f:
            data = json.load(f)
            for key in required_keys:
                if key not in data:
                     logging.error(KeyError(f"Key '{key}' not found in JSON file: {file}"))

    return json_files


def modify_controlnet_images(directory, new_path):
    json_files = find_json_files(directory)
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)

        modified_paths = []
        for path in data["controlnet_images"]:
            modified_path = path.replace("/home/lh/Documents/Database/themes", new_path)
            modified_paths.append(modified_path)

        data["controlnet_images"] = modified_paths

        with open(file, 'w') as f:
            json.dump(data, f, indent=4)