import os.path
import unittest

import cv2

from dedocutils.preprocessing import SkewCorrector


class TestCodeFormat(unittest.TestCase):

    def test_text_detection(self):
        file_path = os.path.join("../data", "document_example.png")
        skew_corrector = SkewCorrector()
        img = skew_corrector.preprocess(cv2.imread(file_path))
        self.assertIsNotNone(img)
