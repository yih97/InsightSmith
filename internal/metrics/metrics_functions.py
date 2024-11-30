"""
This module provides various image quality metrics including IS, FID, LPIPS, SSIM, MSSIM, and PSNR.
"""

import gc
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tf_keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.stats import entropy
from scipy.linalg import sqrtm
import torch
import lpips
from PIL import Image
import cv2
import tf_keras.backend as K

def load_inception_model():
    """
    Loads the InceptionV3 model for feature extraction.

    Example:
     model = load_inception_model()
    """
    return InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

def clear_model_session(model):
    """
    Clears the Keras session to free up memory.

    Example:
     model = load_inception_model()
     clear_model_session(model)
    """
    if model is not None:
        del model
    K.clear_session()
    gc.collect()

def calculate_is(images: List[Image.Image], n_split: int = 1,
                 model: InceptionV3 = load_inception_model()) -> Tuple[float, float]:
    """
    Calculates the Inception Score (IS) for a set of images.

    Args:
    images (List[Image.Image]): List of PIL images to calculate IS for.
    n_split (int): Number of splits.
    model (InceptionV3): Preloaded InceptionV3 model.

    Returns:
    Tuple[float, float]: Mean and standard deviation of the Inception Score.

    Example:
     images = [Image.open('path/to/image1.jpg'), Image.open('path/to/image2.jpg')]
     mean_is, std_is = calculate_is(images, model=load_inception_model())
     print(f"Inception Score: {mean_is} ± {std_is}")
    """

    def get_predictions(images: np.ndarray) -> np.ndarray:
        processed_images = preprocess_input(images)
        predictions = model.predict(processed_images)
        return predictions

    def calculate_kl_divergence(predictions: np.ndarray) -> float:
        p_yx = np.mean(predictions, axis=0)
        scores = [entropy(pyx, p_yx) for pyx in predictions]
        return np.exp(np.mean(scores))

    def load_and_resize_images(images: List[Image.Image]) -> np.ndarray:
        return np.array([tf.image.resize(np.array(img), (299, 299)).numpy() for img in images])

    images_resized = load_and_resize_images(images)
    predictions = get_predictions(images_resized)
    n_part = len(predictions) // n_split
    scores = [calculate_kl_divergence(predictions[i * n_part:(i + 1) * n_part])
              for i in range(n_split)]

    return np.mean(scores), np.std(scores)

def calculate_fid(real_images: np.ndarray, generated_images: np.ndarray,
                  model: InceptionV3 = load_inception_model()) -> float:
    """
    Calculates the Frechet Inception Distance (FID) between two sets of images.

    Args:
    real_images (np.ndarray): Array of real images.
    generated_images (np.ndarray): Array of generated images.
    model (InceptionV3): Preloaded InceptionV3 model.

    Returns:
    float: FID score.

    Example:
     real_images = np.array([np.array(Image.open('path/to/real1.jpg')), np.array(Image.open('path/to/real2.jpg'))])
     generated_images = np.array([np.array(Image.open('path/to/gen1.jpg')), np.array(Image.open('path/to/gen2.jpg'))])
     fid_score = calculate_fid(real_images, generated_images, model=load_inception_model())
     print(f"FID Score: {fid_score}")
    """

    def get_features(images: np.ndarray) -> np.ndarray:
        images = preprocess_input(images)
        features = model.predict(images)
        return features

    real_features = get_features(real_images)
    gen_features = get_features(generated_images)
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_gen))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return fid

def calculate_lpips(real_images, generated_images):
    """
    Calculates the LPIPS score between two sets of images.

    Args:
    real_images (List[Image.Image]): List of real PIL images.
    generated_images (List[Image.Image]): List of generated PIL images.

    Returns:
    float: Average LPIPS score.

    Example:
     real_images = [Image.open('path/to/real1.jpg'), Image.open('path/to/real2.jpg')]
     generated_images = [Image.open('path/to/gen1.jpg'), Image.open('path/to/gen2.jpg')]
     lpips_score = calculate_lpips(real_images, generated_images)
     print(f"LPIPS Score: {lpips_score}")
    """
    loss_fn = lpips.LPIPS(net='alex')

    def preprocess_image(image, size=(224, 224)):
        image = image.convert('RGB')
        image = image.resize(size)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
        return image

    lpips_scores = []
    for real_img, gen_img in zip(real_images, generated_images):
        real_img_tensor = preprocess_image(real_img)
        gen_img_tensor = preprocess_image(gen_img)
        lpips_score = loss_fn(real_img_tensor, gen_img_tensor)
        lpips_scores.append(lpips_score.item())

    return np.mean(lpips_scores)

def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculates the SSIM score between two images.

    Args:
    image1 (np.ndarray): First image.
    image2 (np.ndarray): Second image.

    Returns:
    float: SSIM score.

    Example:
     image1 = np.array(Image.open('path/to/image1.jpg'))
     image2 = np.array(Image.open('path/to/image2.jpg'))
     ssim_score = calculate_ssim(image1, image2)
     print(f"SSIM Score: {ssim_score}")
    """
    return tf.image.ssim(image1, image2, max_val=255).numpy()

def calculate_mssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculates the MSSIM score between two images.

    Args:
    image1 (np.ndarray): First image.
    image2 (np.ndarray): Second image.

    Returns:
    float: MSSIM score.

    Example:
     image1 = np.array(Image.open('path/to/image1.jpg'))
     image2 = np.array(Image.open('path/to/image2.jpg'))
     mssim_score = calculate_mssim(image1, image2)
     print(f"MSSIM Score: {mssim_score}")
    """
    return tf.image.ssim_multiscale(image1, image2, max_val=255).numpy()

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> Union[float, str]:
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
    img1 (np.ndarray): First image (original image).
    img2 (np.ndarray): Second image (image to compare against).

    Returns:
    Union[float, str]: PSNR value between the two images in dB, or 'inf' if the MSE is zero.

    Example:
     img1 = np.array(Image.open('path/to/image1.jpg'))
     img2 = np.array(Image.open('path/to/image2.jpg'))
     psnr_score = calculate_psnr(img1, img2)
     print(f"PSNR Score: {psnr_score}")
    """
    if img1.shape != img2.shape:
        # Suppress pylint warnings for cv2 functions
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr


def load_images(images: List[Image.Image], target_size: Tuple[int, int] = (299, 299)) -> np.ndarray:
    """
    Loads and preprocesses images from given PIL.Image objects.

    Args:
    images (List[Image.Image]): List of PIL.Image objects.
    target_size (Tuple[int, int]): Target size for resizing images.

    Returns:
    np.ndarray: Array of preprocessed images.

    Example:
     images = [Image.open('path/to/image1.jpg'), Image.open('path/to/image2.jpg')]
     images_np = load_images(images)
     print(images_np.shape)
    """
    image_arrays = []
    for image in images:
        image = image.resize(target_size)
        image_array = np.array(image)
        image_arrays.append(image_array)
    return np.array(image_arrays)

if __name__ == "__main__":
    # Load example images (replace with actual image paths)
    inception_model = None
    real_images = [
        Image.open("/home/lh/Documents/VisionForge/internal/metrics/test_img/real/6.jpg"),
        Image.open("/home/lh/Documents/VisionForge/internal/metrics/test_img/real/7.jpg")
    ]
    generated_images = [
        Image.open("/home/lh/Documents/VisionForge/internal/metrics/test_img/gen/1651.png"),
        Image.open("/home/lh/Documents/VisionForge/internal/metrics/test_img/gen/1628.png")
    ]

    try:
        # Load Inception model
        inception_model = load_inception_model()

        # Calculate all scores
        # Inception Score (IS)
        mean_is, std_is = calculate_is(generated_images, model=inception_model)
        print(f"Inception Score (IS): {mean_is:.4f} ± {std_is:.4f}")

        # Frechet Inception Distance (FID)
        real_images_np = load_images(real_images)
        generated_images_np = load_images(generated_images)
        fid_score = calculate_fid(real_images_np, generated_images_np, model=inception_model)
        print(f"Frechet Inception Distance (FID): {fid_score:.4f}")

        # LPIPS
        lpips_score = calculate_lpips(real_images, generated_images)
        print(f"Learned Perceptual Image Patch Similarity (LPIPS): {lpips_score:.4f}")

        # SSIM and MSSIM
        ssim_scores = [calculate_ssim(real_img, gen_img)
                       for real_img, gen_img in zip(real_images_np, generated_images_np)]
        mssim_scores = [calculate_mssim(real_img, gen_img)
                        for real_img, gen_img in zip(real_images_np, generated_images_np)]
        mean_ssim = np.mean(ssim_scores)
        mean_mssim = np.mean(mssim_scores)
        print(f"Structural Similarity Index (SSIM): {mean_ssim:.4f}")
        print(f"Multi-Scale Structural Similarity Index (MSSIM): {mean_mssim:.4f}")

        # PSNR
        psnr_scores = [calculate_psnr(real_img, gen_img)
                       for real_img, gen_img in zip(real_images_np, generated_images_np)]
        mean_psnr = np.mean(psnr_scores)
        print(f"Peak Signal-to-Noise Ratio (PSNR): {mean_psnr:.4f}")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Clear model session
        clear_model_session(inception_model)

        # Check if model object still exists in memory (for debugging)
        if inception_model in locals():
            print("Model object still exists in memory.")
        else:
            print("Model object has been deleted.")
