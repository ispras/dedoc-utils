import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from dedocutils.data_structures.bbox import BBox
from dedocutils.text_detection.abstract_text_detector import AbstractTextDetector
from dedocutils.text_detection.doctr_text_detector.doctr.models import detection_predictor
from dedocutils.text_detection.doctr_text_detector.doctr.models.detection.predictor import DetectionPredictor


class DoctrTextDetector(AbstractTextDetector):

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 on_gpu: bool = False,
                 with_vertical_text_detection: bool = False,
                 arch: Optional[str] = None) -> None:
        self._set_device(on_gpu)
        self._net = None
        self.checkpoint_path = os.path.abspath(checkpoint_path) if checkpoint_path else None
        self.with_vertical_text_detection = with_vertical_text_detection

        default_arch = "db_resnet50_rotation" if with_vertical_text_detection else "db_resnet50"
        self.arch = arch if arch else default_arch

    def detect(self, image: np.ndarray, parameters: Optional[dict] = None) -> List[BBox]:
        return self.detect_with_confidence(image)[0]

    def detect_with_confidence(self, image: np.ndarray) -> Tuple[List[BBox], List[float]]:
        """
        :param image: input image with some text on it
        :return: text coordinates prediction and confidence of prediction
        """
        h, w, _ = image.shape
        boxes, confs = [], []
        batch_preds = self.net([image])
        for pred in batch_preds[0]:
            box = BBox.from_two_points(top_left=(int(pred[0] * w), int(pred[1] * h)), bottom_right=(int(pred[2] * w), int(pred[3] * h)))
            boxes.append(box)
            confs.append(pred[4])

        return boxes, confs

    @property
    def net(self) -> DetectionPredictor:
        """
        Predict consists of the list of bounding box with a format: (left, top, right, bottom, confidence)

        :return: Text Localization model
        """
        # lazy loading net
        if self._net:
            return self._net

        if self.checkpoint_path is None:
            self._net = detection_predictor(arch=self.arch, pretrained=True).eval().to(self.device)
        else:
            self._net = detection_predictor(arch=self.arch, pretrained=False).eval().to(self.device)
            self._net.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.location))

        return self._net

    def _set_device(self, on_gpu: bool) -> None:
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.location = lambda storage, loc: storage.cuda()
        else:
            self.device = torch.device("cpu")
            self.location = "cpu"
