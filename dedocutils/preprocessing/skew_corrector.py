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
        self._step = 1  # step
        self._max_angle = 45  # max angle
        self._min_side = 1000  # the fine sweep runs on an image downscaled to this long side (never upscales a small page)
        self._coarse_side = 512  # the coarse guess runs on this smaller thumbnail

    def preprocess(self, image: np.ndarray, parameters: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        parameters = {} if parameters is None else parameters
        orientation_angle = parameters.get("orientation_angle", 0)

        if orientation_angle:
            image = np.rot90(image, orientation_angle // 90)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        scale = min(1.0, self._min_side / max(thresh.shape[:2]))
        if scale < 1.0:
            thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        coarse_scale = min(1.0, self._coarse_side / max(thresh.shape[:2]))
        thumb = cv2.resize(thresh, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_AREA) if coarse_scale < 1.0 else thresh

        coarse_angles = np.arange(-self._max_angle, self._max_angle + 1, 3)
        coarse = float(coarse_angles[int(np.argmax([self._score(thumb, angle) for angle in coarse_angles]))])
        lo, hi = max(coarse - 4, -self._max_angle), min(coarse + 4, self._max_angle)
        fine_angles = np.arange(lo, hi + 0.001, self._step)
        best_angle = float(fine_angles[int(np.argmax([self._score(thresh, angle) for angle in fine_angles]))])

        rotated = image if best_angle == 0 else rotate_image(image, best_angle)
        return rotated, {"rotated_angle": float(orientation_angle + best_angle)}

    @staticmethod
    def _score(arr: np.ndarray, angle: float) -> float:
        data = rotate_image(arr, angle)
        histogram = np.sum(data, axis=1, dtype=float)
        return float(np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float))
