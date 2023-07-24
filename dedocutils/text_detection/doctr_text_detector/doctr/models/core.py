from typing import Any, Dict, Optional

from dedocutils.text_detection.doctr_text_detector.doctr.utils.repr import NestedObject

__all__ = ['BaseModel']


class BaseModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg
