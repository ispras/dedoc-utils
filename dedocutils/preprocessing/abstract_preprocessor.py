from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractPreprocessor(ABC):
    """
    Abstract class for any preprocessor of the document image.
    It may be used to improve the following recognition results.
    """

    @abstractmethod
    def preprocess(self, image: np.ndarray, parameters: Optional[dict] = None) -> np.ndarray:
        """
        :param image: document image for preprocessing
        :param parameters: some preprocessing parameters
        :return: preprocessed image
        """
        pass
