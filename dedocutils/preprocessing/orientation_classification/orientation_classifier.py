import logging
import warnings

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import resize

from dedocutils.preprocessing.orientation_classification.model import ClassificationModelTorch

logger = logging.getLogger()


class OrientationClassifier:
    """
    Class Classifier for work with Orientation Network. This class set device,
    preprocessing (transform) input data, weights of model
    """
    def __init__(self, checkpoint_path: str) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.location = lambda storage, loc: storage.cuda()
        else:
            self.device = torch.device("cpu")
            self.location = "cpu"

        logger.warning(f"Classifier is set to device {self.device}")

        self.transform = v2.Compose([
            v2.Lambda(_image_resize),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.checkpoint_path = checkpoint_path
        self.classes = [0, 90, 180, 270]
        self._net = None

    @property
    def net(self) -> ClassificationModelTorch:
        if self._net:
            return self._net

        self._net = ClassificationModelTorch(self.checkpoint_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._net.load_state_dict(torch.load(self.checkpoint_path, map_location=self.location))
            logger.info(f"Weights were loaded from {self.checkpoint_path}")

        self._net.to(self.device)
        self._net.eval()
        return self._net

    def predict(self, images: list[Image], batch_size: int = 32) -> list[int]:
        """
        Predict class orientation of input image
        """
        all_orientation_predicted = []

        with torch.no_grad():
            tensor_images = torch.stack(self.transform(images)).float().to(self.device)

            for i in range(0, len(tensor_images), batch_size):
                batch = tensor_images[i:i + batch_size]
                outputs = self.net(batch)

                # first 2 classes mean columns number
                # last 4 classes mean orientation
                _, orientation_out = outputs[:, :2], outputs[:, 2:]
                _, orientation_predicted = torch.max(orientation_out, 1)
                all_orientation_predicted.append(orientation_predicted)

        all_orientation_predicted = torch.cat(all_orientation_predicted, dim=0)
        predicted_angles = [self.classes[int(predicted_angle)] for predicted_angle in all_orientation_predicted]
        logger.info(f"Predicted orientation: {predicted_angles}")
        return predicted_angles


def _image_resize(image: Image) -> Image:
    max_dim = max(image.size)
    image1 = resize(image, size=[round(image.size[1] / max_dim * 1200), round(image.size[0] / max_dim * 1200)])
    white_image = Image.new(size=(1200, 1200), color=(255, 255, 255), mode="RGB")
    white_image.paste(image1)
    return white_image
