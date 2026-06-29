from typing import Optional

import torch
from torch import nn
from torchvision import models


class ClassificationModelTorch(nn.Module):
    """
    Wrapper for EfficientNet B0 model.
    """
    def __init__(self, model_path: Optional[str], num_classes: int = 6) -> None:
        """
        First 2 classes are the number of columns on the page [1 column, 2 columns].
        Last 4 classes are the page orientation in degrees [0, 90, 180, 270].
        """
        super(ClassificationModelTorch, self).__init__()
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=model_path is None)
        self.efficientnet_b0.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.efficientnet_b0(x)
        return out
