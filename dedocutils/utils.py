from typing import Tuple

import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float, color_bound: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=color_bound)
    return rotated_mat
