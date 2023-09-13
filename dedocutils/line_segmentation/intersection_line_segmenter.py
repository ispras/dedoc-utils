from typing import List, Optional, Tuple

import numpy as np

from dedocutils.data_structures.bbox import BBox
from dedocutils.line_segmentation.abstract_line_segmenter import AbstractLineSegmenter


class IntersectionLineSegmenter(AbstractLineSegmenter):
    """
    Line segmentation based on the bboxes y coordinates intersection.
    """
    def __init__(self, intersection_thr: float = 0.1) -> None:
        self.intersection_thr = intersection_thr

    def segment(self, bboxes: List[BBox], parameters: Optional[dict] = None) -> List[List[BBox]]:
        box_group = [[[b.x_top_left, b.y_top_left], [b.x_bottom_right, b.y_bottom_right]] for b in bboxes]
        lines = []
        if len(box_group) > 0:
            # sort bounding boxes into lines
            box_group, lines = self.__segment_lines(np.array(box_group))

        res_lines = []
        for line in lines:
            res_lines.append([BBox(box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]) for box in line])

        return res_lines

    def __segment_lines(self, box_group: np.ndarray) -> Tuple[np.ndarray, List]:
        box_group = box_group[np.argsort(box_group[:, 0, 1])]
        sorted_box_group = np.zeros(box_group.shape)
        lines = []

        # list of indexes
        temp = []
        i = 0

        # check if there is more than one box in the box_group
        if len(box_group) <= 1:
            # since there is only one box in the box group do nothing but copying the box
            return box_group, lines

        while i < len(box_group):
            for j in range(i + 1, len(box_group)):
                if self.__is_on_same_line(box_group[i], box_group[j]):
                    if i not in temp:
                        temp.append(i)

                    if j not in temp:
                        temp.append(j)

            # append temp with i if the current box (i) is not on the same line with any other box
            if len(temp) == 0:
                temp.append(i)

            # put boxes on same line into lined_box_group array
            lined_box_group = box_group[np.array(temp)]
            # sort boxes by startX value
            lined_box_group = lined_box_group[np.argsort(lined_box_group[:, 0, 0])]
            lines.append(lined_box_group)

            # skip to the index of the box that is not on the same line
            i = temp[-1] + 1
            # clear list of indexes
            temp = []

        return sorted_box_group, lines

    def __is_on_same_line(self, box_one: np.ndarray, box_two: np.ndarray) -> float:
        box_one_start_y = box_one[0, 1]
        box_one_end_y = box_one[1, 1]
        box_two_start_y = box_two[0, 1]
        box_two_end_y = box_two[1, 1]

        intersection = max(min(box_one_end_y, box_two_end_y) - max(box_one_start_y, box_two_start_y), 0)
        if intersection == 0:
            return False

        union = min(box_one_end_y - box_one_start_y, box_two_end_y - box_two_start_y)
        return intersection / union > self.intersection_thr
