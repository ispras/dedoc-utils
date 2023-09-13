from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class AbstractPreprocessor(ABC):
    """
    Abstract class for any preprocessor of the document image.
    It may be used to improve the following recognition results.
    """

    @abstractmethod
    def preprocess(self, image: np.ndarray, parameters: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        :param image: document image for preprocessing
        :param parameters: some preprocessing parameters
        :return: preprocessed image and some info parameters
        """
        pass
