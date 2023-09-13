from typing import List, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from dedocutils.data_structures.bbox import BBox
from dedocutils.line_segmentation.abstract_line_segmenter import AbstractLineSegmenter


class ClusteringLineSegmenter(AbstractLineSegmenter):
    """
    Line segmentation based on the bboxes y coordinates clustering.
    """
    def segment(self, bboxes: List[BBox], parameters: Optional[dict] = None) -> List[List[BBox]]:
        heights = [b.height for b in bboxes]
        middles = np.array([(2 * b.y_top_left + b.height) / 2 for b in bboxes])

        eps = np.median(heights) / 2
        dbscan = DBSCAN(eps=eps, min_samples=1)
        pred = dbscan.fit_predict(middles.reshape(-1, 1))

        sorted_bboxes = [dict(bbox=[], avgY=0) for _ in range(max(pred) + 1)]
        for i, pred_item in enumerate(pred):
            bbox = bboxes[i]
            cluster = pred_item
            sorted_bboxes[cluster]["bbox"].append(bbox)
            sorted_bboxes[cluster]["avgY"] += bbox.y_top_left

        for el in sorted_bboxes:
            el["avgY"] /= len(el["bbox"])

        sorted_bboxes = sorted(sorted_bboxes, key=lambda x: x["avgY"])
        sorted_bboxes = [el["bbox"] for el in sorted_bboxes]
        sorted_bboxes = [sorted(line, key=lambda x: x.x_top_left) for line in sorted_bboxes]
        return sorted_bboxes
