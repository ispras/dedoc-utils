from typing import List, Optional

import numpy as np
import pytesseract

from dedocutils.data_structures import BBox
from dedocutils.text_detection import AbstractTextDetector


class TesseractTextDetector(AbstractTextDetector):
    def __init__(self, config: Optional[str] = None) -> None:
        self.config = config if config is not None else "--psm 3"

    def detect(self, image: np.ndarray, parameters: Optional[dict] = None) -> List[BBox]:
        parameters = {} if parameters is None else parameters
        lang = parameters.get("language", "rus+eng")

        data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=self.config)
        left, top, width, height = data["left"], data["top"], data["width"], data["height"]

        bboxes = []
        for x, y, w, h in zip(left, top, width, height):
            bbox = BBox(x_top_left=x, y_top_left=y, width=w, height=h)
            bboxes.append(bbox)

        return bboxes
