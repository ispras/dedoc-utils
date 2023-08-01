import os.path
import unittest

import cv2

from dedocutils.text_detection import DoctrTextDetector, TesseractTextDetector


class TestClasses(unittest.TestCase):

    def test_text_detection(self) -> None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "document_example.png"))
        text_detector = DoctrTextDetector()
        bboxes = text_detector.detect(cv2.imread(file_path))
        self.assertTrue(len(bboxes) > 0)

    @unittest.skip
    def test_tesseract_text_detector(self) -> None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "document_example.png"))
        text_detector = TesseractTextDetector()
        bboxes = text_detector.detect(cv2.imread(file_path))
        self.assertTrue(len(bboxes) > 0)
