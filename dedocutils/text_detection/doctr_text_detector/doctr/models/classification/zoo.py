from typing import Any

from .predictor import CropOrientationPredictor
from .. import classification
from ..preprocessor import PreProcessor

__all__ = ["crop_orientation_predictor"]


ARCHS = ['mobilenet_v3_small_orientation']


def _crop_orientation_predictor(
    arch: str,
    pretrained: bool,
    **kwargs: Any
) -> CropOrientationPredictor:

    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Load directly classifier from backbone
    _model = classification.__dict__[arch](pretrained=pretrained)
    kwargs['mean'] = kwargs.get('mean', _model.cfg['mean'])
    kwargs['std'] = kwargs.get('std', _model.cfg['std'])
    kwargs['batch_size'] = kwargs.get('batch_size', 64)
    input_shape = _model.cfg['input_shape'][1:]
    predictor = CropOrientationPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs),
        _model
    )
    return predictor


def crop_orientation_predictor(
    arch: str = 'mobilenet_v3_small_orientation',
    pretrained: bool = False,
    **kwargs: Any
) -> CropOrientationPredictor:
    """Orientation classification architecture.

    >>> import numpy as np
    >>> from src.utils.utils_models.text_detector.doctr.models import crop_orientation_predictor
    >>> model = crop_orientation_predictor(arch='classif_mobilenet_v3_small', pretrained=True)
    >>> input_crop = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small')
        pretrained: If True, returns a model pre-trained on our recognition crops dataset

    Returns:
        CropOrientationPredictor
    """

    return _crop_orientation_predictor(arch, pretrained, **kwargs)
