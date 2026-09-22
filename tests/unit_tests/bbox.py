import unittest

import numpy as np

from dedocutils.data_structures.bbox import BBox


class TestBBoxBase(unittest.TestCase):

    def test_incorrect_bbox(self) -> None:
        for i in range(4):
            coords = [0, 0, 0, 0]
            coords[i] = -1
            with self.assertRaises(ValueError):
                BBox(*coords)

    def test_bottom_right_properties(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=100, height=50)
        self.assertEqual(bbox.x_bottom_right, 110)
        self.assertEqual(bbox.y_bottom_right, 70)

    def test_square_property(self) -> None:
        bbox = BBox(x_top_left=0, y_top_left=0, width=10, height=5)
        self.assertEqual(bbox.square, 50)

        bbox = BBox(x_top_left=0, y_top_left=0, width=0, height=0)
        self.assertEqual(bbox.square, 0)

    def test_from_two_points(self) -> None:
        bbox = BBox.from_two_points(top_left=(10, 20), bottom_right=(110, 70))
        self.assertEqual(bbox.x_top_left, 10)
        self.assertEqual(bbox.y_top_left, 20)
        self.assertEqual(bbox.width, 100)
        self.assertEqual(bbox.height, 50)

        bbox = BBox.from_two_points(top_left=(5, 5), bottom_right=(5, 5))
        self.assertEqual(bbox.width, 0)
        self.assertEqual(bbox.height, 0)

    def test_shift(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=100, height=50)
        bbox.shift(shift_x=5, shift_y=15)
        self.assertEqual(bbox.x_top_left, 15)
        self.assertEqual(bbox.y_top_left, 35)
        self.assertEqual(bbox.width, 100)
        self.assertEqual(bbox.height, 50)

        bbox = BBox(x_top_left=10, y_top_left=20, width=100, height=50)
        bbox.shift(shift_x=-5, shift_y=-10)
        self.assertEqual(bbox.x_top_left, 5)
        self.assertEqual(bbox.y_top_left, 10)

        bbox = BBox(x_top_left=10, y_top_left=20, width=100, height=50)
        bbox.shift(shift_x=0, shift_y=0)
        self.assertEqual(bbox.x_top_left, 10)
        self.assertEqual(bbox.y_top_left, 20)

    def test_incorrect_shift(self) -> None:
        bbox = BBox(x_top_left=0, y_top_left=0, width=100, height=50)
        with self.assertRaises(ValueError):
            bbox.shift(shift_x=-1, shift_y=-1)


class TestBBoxCropImage(unittest.TestCase):

    def setUp(self) -> None:
        self.image = np.arange(100 * 100).reshape(100, 100)

    def test_crop_simple(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=30, height=40)
        cropped = BBox.crop_image_by_box(self.image, bbox)
        self.assertEqual(cropped.shape, (40, 30))
        expected = self.image[20:60, 10:40]
        np.testing.assert_array_equal(cropped, expected)

    def test_crop_full_image(self) -> None:
        bbox = BBox(x_top_left=0, y_top_left=0, width=100, height=100)
        cropped = BBox.crop_image_by_box(self.image, bbox)
        self.assertEqual(cropped.shape, (100, 100))
        np.testing.assert_array_equal(cropped, self.image)

    def test_crop_zero_size(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=0, height=0)
        cropped = BBox.crop_image_by_box(self.image, bbox)
        self.assertEqual(cropped.shape, (0, 0))


class TestBBoxIntersection(unittest.TestCase):

    def test_identical_boxes(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=100, height=100)
        bbox2 = BBox(x_top_left=0, y_top_left=0, width=100, height=100)
        self.assertTrue(bbox1.have_intersection_with_box(bbox2, threshold=0.99))

    def test_no_intersection(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=100, y_top_left=100, width=10, height=10)
        self.assertFalse(bbox1.have_intersection_with_box(bbox2))

    def test_partial_intersection_above_threshold(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=5, y_top_left=5, width=10, height=10)
        self.assertTrue(bbox1.have_intersection_with_box(bbox2, threshold=0.2))
        self.assertFalse(bbox1.have_intersection_with_box(bbox2, threshold=0.3))

    def test_partial_intersection_exact_threshold(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=5, y_top_left=5, width=10, height=10)
        self.assertFalse(bbox1.have_intersection_with_box(bbox2, threshold=0.25))

    def test_box_b_fully_inside_a(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=100, height=100)
        bbox2 = BBox(x_top_left=10, y_top_left=10, width=20, height=20)
        self.assertTrue(bbox1.have_intersection_with_box(bbox2, threshold=0.99))

    def test_zero_area_box(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=5, y_top_left=5, width=0, height=0)
        self.assertFalse(bbox1.have_intersection_with_box(bbox2))

    def test_touching_edges(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=10, y_top_left=0, width=10, height=10)
        self.assertFalse(bbox1.have_intersection_with_box(bbox2))


class TestBBoxSerialization(unittest.TestCase):

    def test_to_dict(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=30, height=40)
        result = bbox.to_dict()
        self.assertEqual(result["x_top_left"], 10)
        self.assertEqual(result["y_top_left"], 20)
        self.assertEqual(result["width"], 30)
        self.assertEqual(result["height"], 40)

    def test_to_dict_is_ordered(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=30, height=40)
        result = bbox.to_dict()
        self.assertEqual(
            list(result.keys()),
            ["x_top_left", "y_top_left", "width", "height"],
        )

    def test_to_relative_dict(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=30, height=40)
        result = bbox.to_relative_dict(page_width=100, page_height=200)
        self.assertAlmostEqual(result["x_top_left"], 0.1)
        self.assertAlmostEqual(result["y_top_left"], 0.1)
        self.assertAlmostEqual(result["width"], 0.3)
        self.assertAlmostEqual(result["height"], 0.2)
        self.assertEqual(result["page_width"], 100)
        self.assertEqual(result["page_height"], 200)

    def test_from_dict(self) -> None:
        data = {"x_top_left": 10, "y_top_left": 20, "width": 30, "height": 40}
        bbox = BBox.from_dict(data)
        self.assertEqual(bbox.x_top_left, 10)
        self.assertEqual(bbox.y_top_left, 20)
        self.assertEqual(bbox.width, 30)
        self.assertEqual(bbox.height, 40)

    def test_dict_roundtrip(self) -> None:
        original = BBox(x_top_left=1, y_top_left=2, width=3, height=4)
        restored = BBox.from_dict(original.to_dict())
        self.assertEqual(original.x_top_left, restored.x_top_left)
        self.assertEqual(original.y_top_left, restored.y_top_left)
        self.assertEqual(original.width, restored.width)
        self.assertEqual(original.height, restored.height)
        self.assertEqual(original, restored)


class TestBBoxComparison(unittest.TestCase):

    def test_equal(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        self.assertEqual(bbox1, bbox2)
        self.assertEqual(bbox1.__hash__(), bbox2.__hash__())

    def test_less_by_y(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=0, y_top_left=5, width=10, height=10)
        self.assertTrue(bbox1 < bbox2)
        self.assertFalse(bbox2 < bbox1)

    def test_less_by_x_when_y_equal(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        bbox2 = BBox(x_top_left=5, y_top_left=0, width=10, height=10)
        self.assertTrue(bbox1 < bbox2)
        self.assertFalse(bbox2 < bbox1)

    def test_less_by_height_when_xy_equal(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=10, height=5)
        bbox2 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        self.assertTrue(bbox1 < bbox2)

    def test_less_by_width_when_others_equal(self) -> None:
        bbox1 = BBox(x_top_left=0, y_top_left=0, width=5, height=10)
        bbox2 = BBox(x_top_left=0, y_top_left=0, width=10, height=10)
        self.assertTrue(bbox1 < bbox2)

    def test_equal_boxes_not_less(self) -> None:
        bbox1 = BBox(x_top_left=1, y_top_left=2, width=3, height=4)
        bbox2 = BBox(x_top_left=1, y_top_left=2, width=3, height=4)
        self.assertFalse(bbox1 < bbox2)
        self.assertFalse(bbox2 < bbox1)

    def test_sorting(self) -> None:
        bboxes = [
            BBox(x_top_left=5, y_top_left=0, width=1, height=1),
            BBox(x_top_left=0, y_top_left=10, width=1, height=1),
            BBox(x_top_left=0, y_top_left=0, width=1, height=1),
        ]
        sorted_bboxes = sorted(bboxes)
        self.assertEqual(sorted_bboxes[0].y_top_left, 0)
        self.assertEqual(sorted_bboxes[0].x_top_left, 0)
        self.assertEqual(sorted_bboxes[1].y_top_left, 0)
        self.assertEqual(sorted_bboxes[1].x_top_left, 5)
        self.assertEqual(sorted_bboxes[2].y_top_left, 10)


class TestBBoxRotateCoordinates(unittest.TestCase):

    def test_rotate_zero_degrees(self) -> None:
        bbox = BBox(x_top_left=10, y_top_left=20, width=30, height=40)
        image_shape = (100, 100)  # (height, width)
        bbox.rotate_coordinates(angle_rotate=0, image_shape=image_shape)
        self.assertEqual(bbox.x_top_left, 10)
        self.assertEqual(bbox.y_top_left, 20)
        self.assertEqual(bbox.width, 30)
        self.assertEqual(bbox.height, 40)

    def test_rotate_360_degrees(self) -> None:
        bbox = BBox(x_top_left=0, y_top_left=0, width=100, height=100)
        image_shape = (100, 100)
        bbox.rotate_coordinates(angle_rotate=360, image_shape=image_shape)
        self.assertEqual(bbox.x_top_left, 0)
        self.assertEqual(bbox.y_top_left, 0)
        self.assertAlmostEqual(bbox.width, 100, delta=1)
        self.assertAlmostEqual(bbox.width, 100, delta=1)

    def test_rotate_90_degrees(self) -> None:
        bbox = BBox(x_top_left=40, y_top_left=40, width=20, height=20)
        image_shape = (100, 100)
        bbox.rotate_coordinates(angle_rotate=180, image_shape=image_shape)

        self.assertGreaterEqual(bbox.x_top_left, 0)
        self.assertGreaterEqual(bbox.y_top_left, 0)
        self.assertLessEqual(bbox.x_top_left + bbox.width, image_shape[1])
        self.assertLessEqual(bbox.y_top_left + bbox.height, image_shape[0])

    def test_rotate_out_of_image_bounds(self) -> None:
        bbox = BBox(x_top_left=0, y_top_left=0, width=100, height=100)
        image_shape = (100, 100)
        with self.assertRaises(ValueError):
            bbox.rotate_coordinates(angle_rotate=45, image_shape=image_shape)
