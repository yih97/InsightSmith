from typing import List, Union
from internal.components.deepface.deepface import DeepFace
from internal.components.utils import convert_image_to_numpy, convert_numpy_to_image
import cv2
import logging
import numpy as np
from PIL import Image


def face_detection_and_psnr(
        images: List[Image.Image],
        align: bool = True,
        multi_face: bool = False,
        expand_percentage: int = 20,
        model_name: str = 'retinaface'
) -> Union[float, str, None]:
    """
    입력 이미지에서 얼굴을 감지하고, 감지된 얼굴 간의 PSNR (Peak Signal-to-Noise Ratio) 값을 계산합니다.

    Args:
        images (List[Image.Image]): 두 개의 PIL 이미지 객체를 포함하는 리스트.
        align (bool): 얼굴을 정렬할지 여부 (기본값: True).
        multi_face (bool): 여러 얼굴을 감지할지 여부 (기본값: False).
        expand_percentage (int): 감지된 얼굴 영역을 확장할 비율 (기본값: 20).
        model_name (str): 얼굴 감지 백엔드 모델. 옵션: 'retinaface', 'mtcnn' 등 (기본값: 'retinaface').

    Returns:
        Union[float, str, None]: 두 이미지 간의 PSNR 값(dB 단위) 또는 오류 메시지.
    """

    if len(images) != 2:
        logging.error("이미지 목록에는 정확히 두 개의 이미지가 포함되어야 합니다.")
        return None

    detected_faces = []

    for image in images:
        np_image = convert_image_to_numpy(image)
        try:
            faces = DeepFace.extract_faces(np_image, detector_backend=model_name, align=align,
                                           enforce_detection=True)
            if faces is None:
                logging.error('이미지에서 얼굴을 감지할 수 없습니다.')
                return None

            if len(faces) > 1 and not multi_face:
                logging.warning('이미지에서 여러 얼굴이 감지되었습니다. 감지 점수가 가장 높은 얼굴을 선택합니다.')
                faces = [max(faces, key=lambda x: x['confidence'])]


        except Exception as e:
            logging.error(f'얼굴 감지 중 오류 발생: {e}')
            return None

        for face in faces:
            face = face['face']
            face *= (255.0 / face.max())
            face = face.astype(np.uint8)
            face_img = Image.fromarray(face, "RGB")
            detected_faces.append(face_img)

    if len(detected_faces) != 2:
        logging.error("두 이미지에서 각각 한 개의 얼굴을 정확히 감지하지 못했습니다.")
        return None

    original_face = np.array(detected_faces[0])
    comparison_face = np.array(detected_faces[1])

    # PSNR 계산
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> Union[float, str]:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')

        pixel_max = 255.0
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
        return psnr

    psnr_score = calculate_psnr(original_face, comparison_face)
    return psnr_score


# Example usage
if __name__ == "__main__":
    original_image = Image.open('/home/lh/Documents/test/real/w2/27964.png')
    comparison_image = Image.open('/home/lh/Documents/test/gen/w2/20240610183620_enhanced_0.png')

    psnr_score = face_detection_and_psnr([original_image, comparison_image])
    if psnr_score is not None:
        print(f"PSNR Score: {psnr_score}")
    else:
        print("Face detection failed or an error occurred.")
