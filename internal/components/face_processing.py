from typing import List, Dict, Any


from internal.components.deepface.deepface import DeepFace
from internal.components.utils import convert_image_to_numpy, convert_numpy_to_image
import cv2
import logging
import numpy as np
import torch
from PIL import Image
from internal.components.utils import image_to_tensor, hwc2bchw, bchw2hwc
import facer


def face_detection(images: [Image.Image],
                   align: bool = True,
                   multi_face: bool = False,
                   expand_percentage: int = 20,
                   target_size: tuple[int, int] = (512, 512),
                   model_name: str = 'retinaface') -> list[Image.Image] | None:
    """Performs face detection on the input image and returns the cropped faces with or without alignment.

    Example:
        Here's how you can use this function:

        ```
        directory = 'path/to/directory/containing/images'
        image_files = [file for file in os.listdir(directory) if file.endswith('.jpg')]
        detected_faces = face_detection([Image.open(os.path.join(directory, file)) for file in image_files])
        for face in detected_faces:
            face.show()
        ```

    Args:
        images (Image): Accepts a list of images as a PIL Image.
        align (bool): Whether to align the faces or not (default is True).
        multi_face (bool): Whether to detect multiple faces in the image or not
            (default is False),
        model_name (str): face detector backend. Options: 'opencv',
            'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'
            (default is retinaface).
        expand_percentage (int): Percentage to expand the bounding box
            (default is 20).
        target_size (tuple): Size of the target image (default is (224, 224)).

    Returns:
        detected_faces (List[Images]) | None: List of PIL Image objects of the detected faces

    """

    detected_faces = []

    for image in images:
        np_image = convert_image_to_numpy(image)
        try:
            faces = DeepFace.extract_faces(np_image, detector_backend=model_name, align=align,
                                           expand_percentage=expand_percentage, target_size=target_size)
            if faces is None:
                logging.error('No faces detected in the image')
                return None

            if len(faces) > 1 and not multi_face:
                logging.warning('More than one face detected in the image. Keeping the face with highest detection '
                                'score.')
                faces = [max(faces, key=lambda x: x['confidence'])]

        except Exception as e:
            logging.error(f'Error in detecting faces: {e}')
            return None

        for face in faces:
            face = face['face']
            face *= (255.0 / face.max())
            face = face.astype(np.uint8)
            face_img = Image.fromarray(face, "RGB")
            face_img.show()
            detected_faces.append(face_img)

    return detected_faces


def face_recognition(
        target_imgs: [Image.Image],
        src_img: Image.Image,
        model_name: str = 'VGG-Face',
        distance_metric: str = 'cosine',
        enforce_detection: bool = True,
        detector_backend: str = 'retinaface') -> list[dict[str, Any]] | None:
    """ Performs face recognition on the input images, given a source image and returns the verified faces with their
    respective distance and threshold.

    Example:

        Here's how you can use this function:

        ```
        directory = 'path/to/directory/containing/images'
        target_image_files = [file for file in os.listdir(directory) if file.endswith('.jpg')] # Images to verify
        src_image = Image.open(os.path.join(directory, 'source.jpg')) # Source image to verify against
        verified_faces = face_recognition([Image.open(os.path.join(directory, file)) for file in image_files])

        ```

    Args:
        target_imgs: Accepts a list of images that need to be verified
            as a PIL Image.
        src_img: Accepts a single image as a PIL Image.
        model_name: face recognition model. Options: 'VGG-Face',
            'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace',
            'Dlib' (default is VGG-Face).
        distance_metric: distance metric for face recognition. Options:
            'cosine', 'euclidean', 'euclidean_l2' (default is cosine).
        enforce_detection: Whether to enforce face detection or not
            (default is True).
        detector_backend: face detector backend. Options: 'opencv',
            'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'
            (default is retinaface).

    Returns:
        recognised_faces | None : List of dictionaries containing the verified images, distance and threshold.

    """

    np_src_img = convert_image_to_numpy(src_img)
    np_target_imgs = [convert_image_to_numpy(img) for img in target_imgs]
    recognised_faces = []
    info_needed = ['threshold', 'distance']
    try:

        for target_img in np_target_imgs:
            # TODO: Add support for dynamic thresholding when detecting.
            verified_faces = DeepFace.verify(np_src_img, target_img, model_name=model_name,
                                             distance_metric=distance_metric, enforce_detection=enforce_detection,
                                             detector_backend=detector_backend)

            if verified_faces is None:
                logging.error('No faces detected in the image')
                return None

            new_dict = {key: value for key, value in verified_faces.items() if
                        key in info_needed and verified_faces.get('verified', True)}
            new_dict['image'] = convert_numpy_to_image(target_img)
            recognised_faces.append(new_dict)

        recognised_faces = sorted(recognised_faces, key=lambda x: x['distance'])

    except Exception as e:
        logging.error(f'Error in verifying faces: {e}')
        return None

    return recognised_faces


def calculate_face_parsing(
        image: Image.Image,
        parser_model: str = 'farl/lapa/448',
        detection_model: str = 'retinaface/mobilenet') -> Image.Image | None:
    """
    Performs face parsing(hair,face, neck detection) on the input image and returns the parsed mask.

     Example:
        Here's how you can use this function:

        ```
        image = Image.open('path to image')
        ex = calculate_face_parsing(image)
        ex.show()

        ```


    Args:
        image : Accepts a list of images that need to be verified
            as a PIL Image.
        parser_model : Face parsing model. farl/lapa/448" or farl/celebm/44. Default is 'farl/lapa/448'
        detection_model : Face detection model. retinaface/mobilenet or retinaface/ghostnet.
            Default is 'retinaface/mobilenet'

    Returns:
        mask | None : Binary PIL Image of the parsed mask
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_tensor = hwc2bchw(image_to_tensor(image)).to(device=device)
    face_parser = facer.face_parser(parser_model, device=torch.device(device))
    face_detector = facer.face_detector(detection_model, device=torch.device(device))

    try:
        with torch.inference_mode():
            faces = face_detector(image_tensor)
            parser = face_parser(image_tensor, faces)
            seg_logits = parser['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
            n_classes = seg_probs.size(1)
            vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
            vis_img = vis_seg_probs.sum(0, keepdim=True)
            vis_img_ = bchw2hwc(vis_img.unsqueeze(1))
            if vis_img_.dtype != torch.uint8:
                vis_img_ = vis_img_.to(torch.uint8)
            if vis_img_.size(2) == 1:
                vis_img_ = vis_img_.repeat(1, 1, 3)
            pimage = vis_img_.cpu().numpy()
            mask = Image.fromarray(np.where(pimage > 0, 255, 0).astype(np.uint8)).convert('L')
            return mask
            # mask.show()
    except Exception as e:
        logging.error(f'Error in calculating face parsing: {e}')
        return None



if __name__=="__main__":
    import os
    directory = '/media/hamna/Extra/dataset/People/w1/data/'
    target_image_files = [file for file in os.listdir(directory) if file.endswith('.jpg')]  # Images to verify
    src_image = Image.open(os.path.join(directory, 'source.png'))  # Source image to verify against
    verified_faces = face_recognition([Image.open(os.path.join(directory, file)) for file in target_image_files], src_image)
    print(verified_faces)