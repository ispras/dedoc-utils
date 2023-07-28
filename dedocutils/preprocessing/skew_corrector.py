from typing import Optional, Tuple

import cv2
import numpy as np

from dedocutils.preprocessing.abstract_preprocessor import AbstractPreprocessor


class SkewCorrector(AbstractPreprocessor):
    """
    This class is used for automatic skew correction of the document image.
    It is useful for small angles skew correction.
    The projection method is used to determine the rotation angle.
    """
    def __init__(self) -> None:
        self.step = 1  # step
        self.max_angle = 45  # max angle

    def preprocess(self, image: np.ndarray, parameters: Optional[dict] = None) -> np.ndarray:
        parameters = {} if parameters is None else parameters
        orientation_angle = parameters.get("orientation_angle", 0)

        if orientation_angle:
            image = self.__rotate_image(image, orientation_angle)

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

        rotated = self.__rotate_image(image, best_angle)
        return rotated

    def __determine_score(self, arr: np.ndarray, angle: int) -> (np.ndarray, float):
        data = self.__rotate_image(arr, angle)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score

    def __rotate_image(self, image: np.ndarray, angle: float, color_bound: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
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
