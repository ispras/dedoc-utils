from typing import List, Optional

import numpy as np
import pytesseract

from dedocutils.data_structures import BBox
from dedocutils.data_structures.text_with_bbox import TextWithBBox
from dedocutils.text_detection_recognition.abstract_detector_recognizer import AbstractDetectorRecognizer


class TesseractDetectorRecognizer(AbstractDetectorRecognizer):

    def __init__(self, config: str = "--psm 3") -> None:
        self.config = config

    def detect_recognize(self, image: np.ndarray, parameters: Optional[dict] = None) -> List[TextWithBBox]:
        parameters = {} if parameters is None else parameters
        lang = parameters.get("language", "rus+eng")

        data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=self.config)
        words, left, top, width, height, levels = data["text"], data["left"], data["top"], data["width"], data["height"], data["level"]

        # filter empty words and corresponding coordinates
        irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
        words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
        left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
        height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]
        levels = [level for idx, level in enumerate(levels) if idx not in irrelevant_indices]
        assert len(words) == len(left) == len(top) == len(width) == len(height), "Number of words and their coordinates should be equal"

        text_with_bbox_list = []
        for word, x, y, w, h, level in zip(words, left, top, width, height, levels):
            if level == 5:
                twb = TextWithBBox(text=word, bbox=BBox(x_top_left=x, y_top_left=y, width=w, height=h))
                text_with_bbox_list.append(twb)

        return text_with_bbox_list
