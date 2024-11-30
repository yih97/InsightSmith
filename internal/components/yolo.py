from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
import logging
import os
from dotenv import load_dotenv
load_dotenv()


try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Please install ultralytics using `pip install ultralytics`")
    raise


def create_mask_from_bbox(
    bboxes: np.ndarray, shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill="white")
        masks.append(mask)
    return masks


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def yolo_detector(
    image: Image.Image, model_path: str | Path | None = None, confidence: float = 0.3
) -> list[Image.Image] | None:
    if not model_path:
        model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
    model = YOLO(model_path)
    pred = model(image, conf=confidence)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return None

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    return masks


def yolo_instance_segmentation(
    input_image: Image.Image | list[Image.Image],
    model_path: str = 'yolov8m-seg.pt'
) -> list[Image.Image] | Image.Image | None:

    """
    Perform instance (Person) segmentation on the input image.

    Example:
        Here's how you can use this function for a list of images or a single image

        ```
        img1 = Image.open('path/to/image1')
        img2 = Image.open('path/to/image2')
        masks = yolo_instance_segmentation([img1, img2])
        for mask in masks:
            mask.show()
        ```

        ```
        img = Image.open('path/to/image')
        mask = yolo_instance_segmentation(img)
        mask.show()
        ```


    Args:
        input_image: Image in PIL format
        model_path: Path to the model file. Default is 'yolov8m-seg.pt' which is a pre-trained model.
        Can be any of the models listed at: https://docs.ultralytics.com/tasks/segment/

    Returns:
        Masked images

    """

    if not isinstance(input_image, list):
        input_image = [input_image]

    pred = None
    try:
        model_path = os.getenv('MODEL_ZOO')
        model_path = os.path.join(model_path, 'yolo/yolov8m-seg.pt')
        model = YOLO(model_path)
        pred = model.predict(input_image, classes=0, conf=0.6) #classes= ensures only person class is detected.

    except Exception as e:
        logging.error(f"Error loading instance segmentation model and performing segmentation: {e}")

    if pred is not None:
        try:
            masks = []
            for i,result in enumerate(pred):
                bboxes = result[0].boxes.xyxy.cpu().numpy()
                if result[0].masks is None:
                    mask = create_mask_from_bbox(bboxes, input_image[i].size)[0]
                else:
                    mask = mask_to_pil(result[0].masks.data, input_image[i].size)[0]
                # mask.show()
                masks.append(mask)

            if len(masks) == 1:
                return masks[0]
            else:
                return masks

        except Exception as e:
            logging.error(f"Error creating masks: {e}")
            return None


if __name__ == '__main__':
    input_image1 = Image.open('/home/hamna/Downloads/u90.jpg')
    input_image2 = Image.open('/home/hamna/Downloads/u1.jpg')
    y = yolo_instance_segmentation(input_image1)
    y.show()

    # print(y)