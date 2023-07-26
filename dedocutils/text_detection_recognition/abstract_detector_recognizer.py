from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from dedocutils.data_structures.text_with_bbox import TextWithBBox


class AbstractDetectorRecognizer(ABC):
    """
    Abstract class for both text detection and recognition on the document image.
    It allows to find bounding boxes of textual lines with text.
    """

    @abstractmethod
    def detect_recognize(self, image: np.ndarray, parameters: Optional[dict] = None) -> List[TextWithBBox]:
        """
        Detect text coordinates (bounding box) on the image and recognize text for each bounding box.

        :param image: image to detect text on it
        :param parameters: some parameters for detection and recognition
        :return: list of bounding boxes with text
        """
        pass
