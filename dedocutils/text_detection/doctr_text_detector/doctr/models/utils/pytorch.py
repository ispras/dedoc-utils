import logging
from typing import Any, List, Optional

import torch
from torch import nn

from dedocutils.text_detection.doctr_text_detector.doctr.utils.data import download_from_url

__all__ = ['load_pretrained_params', 'conv_sequence_pt']


def load_pretrained_params(
    model: nn.Module,
    url: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from dedocutils.text_detection.doctr_text_detector.doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the keras model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        overwrite: should the zip extraction be enforced if the archive has already been extracted
    """

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir='models', **kwargs)

        # Read state_dict
        state_dict = torch.load(archive_path, map_location='cpu')

        # Load weights
        model.load_state_dict(state_dict)


def conv_sequence_pt(
    in_channels: int,
    out_channels: int,
    relu: bool = False,
    bn: bool = False,
    **kwargs: Any,
) -> List[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from dedocutils.text_detection.doctr_text_detector.doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs['bias'] = kwargs.get('bias', not(bn))
    # Add activation directly to the conv if there is no BN
    conv_seq: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, **kwargs)
    ]

    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))

    if relu:
        conv_seq.append(nn.ReLU(inplace=True))

    return conv_seq
