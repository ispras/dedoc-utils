from typing import Optional, Tuple

import cv2
import numpy as np

from dedocutils.preprocessing.abstract_preprocessor import AbstractPreprocessor
from dedocutils.utils import rotate_image


class SkewCorrector(AbstractPreprocessor):
    """
    This class is used for automatic skew correction of the document image.
    It is useful for small angles skew correction.
    The projection method is used to determine the rotation angle.
    """
    def __init__(self) -> None:
        self.step = 1  # step
        self.max_angle = 45  # max angle

    def preprocess(self, image: np.ndarray, parameters: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        parameters = {} if parameters is None else parameters
        orientation_angle = parameters.get("orientation_angle", 0)

        if orientation_angle:
            rotation_nums = orientation_angle // 90
            image = np.rot90(image, rotation_nums)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        angles = np.arange(-self.max_angle, self.max_angle + self.step, self.step)
        scores = [self.__determine_score(thresh, angle) for angle in angles]

        max_idx = scores.index(max(scores))
        if max_idx >= 2 and scores[max_idx - 2] > scores[max_idx] * 0.98:
            # if there are 2 approximately equal scores +- 1 step by max_score it will utilize angle between them
            best_angle = angles[max_idx - 1]
        elif max_idx < len(scores) - 2 and scores[max_idx + 2] > scores[max_idx] * 0.98:
            best_angle = angles[max_idx + 1]
        else:
            best_angle = angles[scores.index(max(scores))]

        rotated = rotate_image(image, best_angle)
        return rotated, {"rotated_angle": float(orientation_angle + best_angle)}

    def __determine_score(self, arr: np.ndarray, angle: int) -> Tuple[np.ndarray, float]:
        data = rotate_image(arr, angle)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score
