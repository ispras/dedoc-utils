from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from dedocutils.data_structures.bbox import BBox


class AbstractTextDetector(ABC):
    """
    Abstract class for text detection on the document image.
    It allows to find bounding boxes of textual lines or words.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, parameters: Optional[dict] = None) -> List[BBox]:
        """
        Detect text coordinates on the image.

        :param image: image to detect text on it
        :param parameters: some parameters for detection
        :return: list with bounding boxes of found text
        """
        pass
