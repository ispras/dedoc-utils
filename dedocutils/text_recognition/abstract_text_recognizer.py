from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


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
