import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class BBox:
    """
    Bounding box around some page object, the coordinate system starts from top left corner.
    """
    """

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
    def __init__(self, x_top_left: int, y_top_left: int, width: int, height: int) -> None:
        """
        The following parameters should have values of pixels number.

        :param x_top_left: x coordinate of the bbox top left corner
        :param y_top_left: y coordinate of the bbox top left corner
        :param width: bounding box width
        :param height: bounding box height
        """
        self.x_top_left = x_top_left
        self.y_top_left = y_top_left
        self.width = width
        self.height = height

    @property
    def x_bottom_right(self) -> int:
        return self.x_top_left + self.width

    @property
    def y_bottom_right(self) -> int:
        return self.y_top_left + self.height

    @staticmethod
    def crop_image_by_box(image: np.ndarray, bbox: "BBox") -> np.ndarray:
        return image[bbox.y_top_left:bbox.y_bottom_right, bbox.x_top_left:bbox.x_bottom_right]

    def shift(self, shift_x: int, shift_y: int) -> None:
        self.x_top_left += shift_x
        self.y_top_left += shift_y

    def rotate_coordinates(self, angle_rotate: float, image_shape: Tuple[int]) -> None:
        xb, yb = self.x_top_left, self.y_top_left
        xe, ye = self.x_bottom_right, self.y_bottom_right
        rad = angle_rotate * math.pi / 180

        xc = image_shape[1] / 2
        yc = image_shape[0] / 2

        bbox_xb = min((int(float(xb - xc) * math.cos(rad) - float(yb - yc) * math.sin(rad) + xc)), image_shape[1])
        bbox_yb = min((int(float(yb - yc) * math.cos(rad) + float(xb - xc) * math.sin(rad) + yc)), image_shape[0])
        bbox_xe = min((int(float(xe - xc) * math.cos(rad) - float(ye - yc) * math.sin(rad) + xc)), image_shape[1])
        bbox_ye = min((int(float(ye - yc) * math.cos(rad) + float(xe - xc) * math.sin(rad) + yc)), image_shape[0])
        self.__init__(bbox_xb, bbox_yb, bbox_xe - bbox_xb, bbox_ye - bbox_yb)

    def __str__(self) -> str:
        return f"BBox(x = {self.x_top_left} y = {self.y_top_left}, w = {self.width}, h = {self.height})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def square(self) -> int:
        """
        Square of the bbox.
        """
        return self.height * self.width

    @staticmethod
    def from_two_points(top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> "BBox":
        """
        Make the bounding box from two points.

        :param top_left: (x, y) point of the bbox top left corner
        :param bottom_right: (x, y) point of the bbox bottom right corner
        """
        x_top_left, y_top_left = top_left
        x_bottom_right, y_bottom_right = bottom_right
        return BBox(x_top_left=x_top_left, y_top_left=y_top_left, width=x_bottom_right - x_top_left, height=y_bottom_right - y_top_left)

    def have_intersection_with_box(self, box: "BBox", threshold: float = 0.3) -> bool:
        """
        Check if the current bounding box has the intersection with another one.

        :param box: another bounding box to check intersection with
        :param threshold: the lowest value of the intersection over union used get boolean result
        """
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

    def __hash__(self) -> int:
        return hash((self.x_top_left, self.y_top_left, self.width, self.height))
