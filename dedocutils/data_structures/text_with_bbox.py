from dataclasses import dataclass

from dedocutils.data_structures import BBox


@dataclass
class TextWithBBox:
    text: str
    bbox: BBox
