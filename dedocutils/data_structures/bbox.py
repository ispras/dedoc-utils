from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class BBox:
    """
    Box around some page object, the coordinate system starts from top left corner

    0------------------------------------------------------------------------------------------------> x
    |                                   BBox
    |    (x_top_left, y_top_left)  o_____________________
    |                              |                    |
    |                              |     some text      |
    |                              |____________________o
    |                                                   (x_bottom_right, y_bottom_right)
    |
    |
    |
    |
    V y
    """
    x_top_left: int
    y_top_left: int
    width: int
    height: int

    @property
    def x_bottom_right(self) -> int:
        return self.x_top_left + self.width

    @property
    def y_bottom_right(self) -> int:
        return self.y_top_left + self.height

    @property
    def square(self) -> int:
        return self.height * self.width

    @staticmethod
    def from_two_points(top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> "BBox":
        x_top_left, y_top_left = top_left
        x_bottom_right, y_bottom_right = bottom_right
        return BBox(x_top_left=x_top_left, y_top_left=y_top_left, width=x_bottom_right - x_top_left, height=y_bottom_right - y_top_left)

    def have_intersection_with_box(self, box: "BBox", threshold: float = 0.3) -> bool:
        # determine the (x, y)-coordinates of the intersection rectangle
        x_min = max(self.x_top_left, box.x_top_left)
        y_min = max(self.y_top_left, box.y_top_left)
        x_max = min(self.x_top_left + self.width, box.x_top_left + box.width)
        y_max = min(self.y_top_left + self.height, box.y_top_left + box.height)
        # compute the area of intersection rectangle
        inter_a_area = max(0, x_max - x_min) * max(0, y_max - y_min)
        # compute the area of both the prediction and ground-truth
        # rectangles
        box_b_area = float(box.width * box.height)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        percent_intersection = inter_a_area / box_b_area if box_b_area > 0 else 0
        # return the intersection over union value
        return percent_intersection > threshold

    def to_dict(self) -> dict:
        res = OrderedDict()
        res["x_top_left"] = self.x_top_left
        res["y_top_left"] = self.y_top_left
        res["width"] = self.width
        res["height"] = self.height
        return res

    def to_relative_dict(self, page_width: int, page_height: int) -> dict:
        res = OrderedDict()
        res["x_top_left"] = self.x_top_left / page_width
        res["y_top_left"] = self.y_top_left / page_height
        res["width"] = self.width / page_width
        res["height"] = self.height / page_height
        res["page_width"] = page_width
        res["page_height"] = page_height
        return res

    @staticmethod
    def from_dict(some_dict: Dict[str, int]) -> "BBox":
        return BBox(**some_dict)
