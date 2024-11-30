import numpy as np
from typing import Tuple, Callable
from internal.components.utils import resize_image_with_pad
from PIL import Image

class OpenposeModel(object):
    def __init__(self) -> None:
        self.model_openpose = None

    def run_model(
            self,
            img: np.ndarray,
            include_body: bool,
            include_hand: bool,
            include_face: bool,
            use_dw_pose: bool = False,
            use_animal_pose: bool = False,
            json_pose_callback: Callable[[str], None] = None,
            res: int = 512,
            **kwargs  # Ignore rest of kwargs
    ) -> Tuple[np.ndarray, bool]:
        """Run the openpose model. Returns a tuple of
        - result image
        - is_image flag

        The JSON format pose string is passed to `json_pose_callback`.
        """
        if json_pose_callback is None:
            json_pose_callback = lambda x: None

        img, remove_pad = resize_image_with_pad(img, res)

        if self.model_openpose is None:
            from internal.components.openpose import OpenposeDetector
            self.model_openpose = OpenposeDetector()

        return remove_pad(self.model_openpose(
            img,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            use_dw_pose=use_dw_pose,
            use_animal_pose=use_animal_pose,
            json_pose_callback=json_pose_callback
        )), True

    def unload(self):
        if self.model_openpose is not None:
            self.model_openpose.unload_model()


def calculate_openpose(
        input_img: Image.Image,
        include_body: bool = True,
        include_hand: bool = False,
        include_face: bool = False,
        use_dw_pose: bool = False,
        use_animal_pose: bool = False,
        json_pose_callback: Callable[[str], None] = None
) -> Image.Image:

    """ Performs pose estimation on the input image and returns the pose map. The pose map is a visual representation of
    the pose of the person in the image.

    Example:
        Here's how you can use this function:

        ```
        img = Image.open('path/to/image')
        pose = calculate_openpose(img)
        pose.show()
        ```

    Args:
        input_img: Image in PIL format
        include_body: Include body in the output
        include_hand: Include hand in the output
        include_face: Include face in the output
        use_dw_pose: Use dynamic weight pose
        use_animal_pose: Use animal pose
        json_pose_callback: Callback function for JSON pose

    Returns:
        image in PIL format
    """

    resolution = input_img.width
    numpy_img = np.array(input_img)

    openpose_model = OpenposeModel()
    openpose_image, is_image = openpose_model.run_model(
        numpy_img,
        include_body=include_body,
        include_hand=include_hand,
        include_face=include_face,
        use_dw_pose=use_dw_pose,
        use_animal_pose=use_animal_pose,
        json_pose_callback=json_pose_callback,
        res=resolution
    )
    openpose_model.unload()
    return Image.fromarray(openpose_image, "RGB")


