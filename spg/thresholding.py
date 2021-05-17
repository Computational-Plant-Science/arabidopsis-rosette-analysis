import numpy as np
import cv2

def simple_threshold(image: np.ndarray, threshold: int = 90, invert: bool = False) -> np.ndarray:
    if threshold < 0 or threshold > 255:
        raise ValueError(f"Threshold must be between 0 and 255")

    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
    return image


def adaptive_threshold_mean(image: np.ndarray, invert: bool = False) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        81,
        1)


def adaptive_threshold_gaussian(image: np.ndarray, invert: bool = False) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        81,
        1)


def otsu_threshold(image: np.ndarray, invert: bool = False) -> np.ndarray:
    image = cv2.createCLAHE(clipLimit=3).apply(image)
    _, image = cv2.threshold(image, 0, 255, (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
    return image