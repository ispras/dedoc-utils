from abc import ABC, abstractmethod
from typing import List, Optional

from dedocutils.data_structures.bbox import BBox


class AbstractLineSegmenter(ABC):
    """
    The abstract class for sorting detected words on the image to get document lines.
    It uses the output from a text detector which returns bounding boxes of words.
    """

    @abstractmethod
    def segment(self, bboxes: List[BBox], parameters: Optional[dict] = None) -> List[List[BBox]]:
        """
        :param bboxes: list of words bboxes detected by a text detector
        :param parameters: some segmentation parameters
        :return: list of lines, each contains list of sorted word bboxes
        """
        pass
