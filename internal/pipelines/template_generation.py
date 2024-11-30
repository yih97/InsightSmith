import logging

import numpy as np
from PIL import Image
from typing import List
from internal.components.edge_detection import calculate_canny, calculate_softedge
from internal.components.pose_estimation import calculate_openpose
from internal.components.yolo import yolo_instance_segmentation
from internal.components.face_processing import calculate_face_parsing
import cv2


def dilute_mask(mask: np.ndarray, dilution_factor: int) -> np.ndarray:
    """Dilute the mask by a factor of dilution_factor.

    Args:
        mask (np.ndarray): Mask to dilute
        dilution_factor (int): Factor to dilute the mask by

    Returns:
        np.ndarray: Diluted mask
    """
    dilution_factor = max(1, dilution_factor)
    dilated_mask = np.zeros_like(mask)
    for i in range(dilution_factor):
        dilated_mask = np.maximum(dilated_mask, np.roll(mask, i, axis=0))
        dilated_mask = np.maximum(dilated_mask, np.roll(mask, -i, axis=0))
        dilated_mask = np.maximum(dilated_mask, np.roll(mask, i, axis=1))
        dilated_mask = np.maximum(dilated_mask, np.roll(mask, -i, axis=1))
    return dilated_mask


def generate_lora_mask(binary_mask: np.ndarray) -> np.ndarray:
    # Convert black areas to blue
    red_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    red_mask[binary_mask == 0] = [255, 0,0]  # Green color (RGB)

    # Convert white areas to red
    green_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    green_mask[binary_mask == 255] = [0,255,0]  # Red color (RGB)

    # Combine the red and blue masks
    combined_mask = cv2.add(green_mask, red_mask)

    return combined_mask


def generate_lora_mask_v2(person_mask: np.ndarray, face_mask:np.ndarray) -> np.ndarray:

    lora_mask = generate_lora_mask(person_mask)
    lora_mask[face_mask == 255] = [0, 0, 255]  # Blue color (RGB)

    return lora_mask

def preprocess(image: Image.Image | List[Image.Image], dilate_face_mask: bool = True,
               dilate_person_mask: bool = True, num_lora_mask:int =2) -> list[list[Image]]:
    """Preprocess the input image to generate the required templates.

    Example:
        Here's how you can use this function for a list of images or a single image
        ```
        preprocess(Image.open('path/to/image'))
        preprocess([Image.open('path/to/image1'), Image.open('path/to/image2')])
        ```
    Args:
        image (Image.Image): Input image or list of images
        dilate_face_mask (bool): Whether to dilate the face mask or not (default is True)
        dilate_person_mask (bool): Whether to dilate the person mask or not (default is True)
    Returns:
        List[Image.Image]: List of preprocessed images
    """
    if isinstance(image, Image.Image):
        image = [image]

    templates = []

    for input_image in image:
        openpose_template = calculate_openpose(input_image)

        person_mask = np.array(yolo_instance_segmentation(input_image))
        background_mask = np.invert(person_mask)
        face_mask = np.array(calculate_face_parsing(input_image))

        if dilate_face_mask:
            face_mask = dilute_mask(face_mask, 10)

        if dilate_person_mask:
            person_mask = dilute_mask(person_mask, 5)

        canny_image = np.array(calculate_canny(input_image))
        canny_template = Image.fromarray(np.where(person_mask == 0, canny_image, 0))

        softedge_image = np.array(calculate_softedge(input_image))
        softedge_template_noface = np.where(face_mask == 0, softedge_image, 0)
        softedge_template_with_background = Image.fromarray(softedge_template_noface)

        softedge_template_without_background = Image.fromarray(
            np.where(background_mask == 0, softedge_template_noface, 0))

        ##Lora mask Mask generation
        if num_lora_mask == 2:
            lora_mask = Image.fromarray(generate_lora_mask(person_mask))
        elif num_lora_mask == 3:
            lora_mask = Image.fromarray(generate_lora_mask_v2(person_mask,face_mask))
        else:
            lora_mask=None
            logging.error("Invalid number of lora mask requested. Must be 2 or 3")


        templates.append([canny_template, softedge_template_without_background,
                          softedge_template_with_background, openpose_template, lora_mask])

    return templates



if __name__ == "__main__":
    # name = 'newspaper_2'
    # templates = preprocess(Image.open('/home/hamna/Database/themes/newspaper/newspaper_2/newspaper1.png'), num_lora_mask=2)
    # for i, template in enumerate(templates):
    #     for j, image in enumerate(template):
    #         image.save(f'/home/hamna/Database/themes/newspaper/newspaper_2/{name}_{i}_{j}.png')

    img = Image.open('/home/hamna/Database/themes/Military/military_1/loramask.png')
    img = img.convert("RGB")

    # Load the pixel data
    pixels = img.load()

    # Get image dimensions
    width, height = img.size

    # Loop through each pixel and swap the red and blue channels
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            pixels[x, y] = (b, g, r)

    img.save('/home/hamna/Database/themes/Military/military_1/military_2.png')