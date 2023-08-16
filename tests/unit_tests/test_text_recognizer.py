import os.path
import unittest

import cv2

from dedocutils.text_detection import DoctrTextDetector
from dedocutils.text_recognition import TesseractTextRecognizer


class TestTextRecognizer(unittest.TestCase):
    @unittest.skip
    def test_tesseract_recognize(self) -> None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "document_example.png"))
        text_recognizer = TesseractTextRecognizer()
        text = text_recognizer.recognize(cv2.imread(file_path))
        self.assertIn("Document example", text)

    @unittest.skip
    def test_tesseract_recognize_bboxes(self) -> None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "document_example.png"))

        text_detector = DoctrTextDetector()
        text_recognizer = TesseractTextRecognizer()

        img = cv2.imread(file_path)
        bboxes = text_detector.detect(img)
        texts = text_recognizer.recognize_bboxes(img, bboxes)
        self.assertEquals("Document", texts[-1])
