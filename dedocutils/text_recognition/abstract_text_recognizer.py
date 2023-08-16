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

    def recognize_bbox(self, image: np.ndarray, box: BBox, need_rotate: bool = False) -> str:
        line_img = image[box.y_top_left:box.y_bottom_right, box.x_top_left:box.x_bottom_right]

        if need_rotate:
            line_img = np.rot90(line_img)

        line_img = np.pad(line_img, [(15, 15), (15, 15), (0, 0)], constant_values=255)
        text = self.recognize(line_img)
        return text

    def recognize_bboxes(self, image: np.ndarray, bboxes: List[BBox]) -> List[str]:
        return [self.recognize_bbox(image, bbox) for bbox in bboxes]
