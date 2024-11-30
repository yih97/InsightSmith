import logging

import cv2
from PIL import Image
import numpy as np

from internal.components.pidinet import apply_pidinet


def calculate_canny(
        img: Image.Image, low_threshold: int = 50, high_threshold: int = 100
) -> Image.Image | None:
    """Apply Canny edge detection to the image

    Example:
        Here's how you can use this function:

        ```
        img = Image.open('path/to/image')
        canny_img = calculate_canny(img)
        canny_img.show()
        ```

    Arguments:
        img (Image): PIL Image in RGB mode
        low_threshold (float): low threshold for the hysteresis procedure (Default is 50)
        high_threshold (float): high threshold for the hysteresis procedure (Default is 100)

    Returns:
        canny_img (Image): PIL Image in RGB mode

    """

    # Convert PIL image to numpy array
    np_img = np.array(img)[:, :, ::-1]
    try:
        canny_img = cv2.Canny(np_img, low_threshold, high_threshold)
    except Exception as e:
        logging.error(f"Error in applying Canny edge detection: {e}")
        return None
    # Convert numpy array to PIL image
    # canny_img = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
    canny_img = canny_img.astype(np.uint8)
    canny_img = Image.fromarray(canny_img, "L")
    return canny_img


def calculate_softedge(
        img: Image.Image,
        is_safe: bool = True
) -> Image.Image | None:
    """ Use pidinet to calculate soft edge map of an image

    Example:
        Here's how you can use this function:

        ```
        img = Image.open('path/to/image')
        softedge_img = calculate_softedge(img)
        softedge_img.show()
        ```


    Arguments:
        img (Image): Input image in PIL format
        is_safe (bool): Whether to apply the safe version of PIDi-Net or not (default is True)
    Returns:
        softedge_img (Image): Softedge map in PIL format

    """

    # Convert PIL image to numpy array
    np_img = np.array(img)[:, :, ::-1]
    try:
        softedge_img = apply_pidinet(np_img, is_safe=is_safe)
    except Exception as e:
        logging.error(f"Error in applying soft edge detection: {e}")
        return None

    # softedge_img = cv2.cvtColor(softedge, cv2.COLOR_BGR2RGB)
    softedge_img = softedge_img.astype(np.uint8)
    softedge_img = Image.fromarray(softedge_img, "L")
    return softedge_img




