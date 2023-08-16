from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from dedocutils.data_structures import BBox


class AbstractTextRecognizer(ABC):
    """
    Abstract class for OCR on document images.
    """
    @abstractmethod
    def recognize(self, image: np.ndarray, parameters: Optional[dict] = None) -> str:
        """
        Recognize text from the given image.

        :param image: image with a textual line
        :param parameters: some recognition parameters (e.g. language)
        :return: recognized text
        """
        pass

    def recognize_bbox(self, image: np.ndarray, bbox: BBox, need_rotate: bool = False) -> str:
        line_image = image[bbox.y_top_left:bbox.y_bottom_right, bbox.x_top_left:bbox.x_bottom_right]

        if need_rotate:
            line_image = np.rot90(line_image)

        line_image = np.pad(line_image, [(15, 15), (15, 15), (0, 0)], constant_values=255)
        text = self.recognize(line_image)
        return text

    def recognize_bboxes(self, image: np.ndarray, bboxes: List[BBox]) -> List[str]:
        return [self.recognize_bbox(image, bbox) for bbox in bboxes]
