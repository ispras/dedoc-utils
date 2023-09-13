import os.path
import unittest

import cv2

from dedocutils.preprocessing import SkewCorrector


class TestPreprocessors(unittest.TestCase):
    def test_skew_correction(self) -> None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "document_example_rotated.png"))
        skew_corrector = SkewCorrector()
        image = cv2.imread(file_path)
        rotated_image, info = skew_corrector.preprocess(image)

        self.assertEqual(info["rotated_angle"], 5.0)
