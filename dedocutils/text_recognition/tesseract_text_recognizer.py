from typing import Optional

import numpy as np
import pytesseract

from dedocutils.text_recognition.abstract_text_recognizer import AbstractTextRecognizer


class TesseractTextRecognizer(AbstractTextRecognizer):

    def __init__(self, config: str = "--psm 6") -> None:
        self.config = config

    def recognize(self, image: np.ndarray, parameters: Optional[dict] = None) -> str:
        parameters = {} if parameters is None else parameters
        lang = parameters.get("language", "rus+eng")

        text = pytesseract.image_to_string(image, lang=lang, config=self.config)
        return text
