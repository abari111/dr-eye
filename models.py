import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    vgg16,
    VGG16_Weights,
)


class RetNet(nn.Module):
    """Custom network for retinopathy detection"""

    def __init__(self, in_c: int, out_c: int, nb_cl: int = 5):

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_c, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(nn.Linear(246016, 1000), nn.Linear(1000, nb_cl))

    def forward(self, x):

        x = self.features(x)
        x = x.view(-1, 246016)
        x = self.classifier(x)
        return x


def get_mobilenet_v3(
    n_classes: int = 5, weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
) -> torch.nn.Module:
    """Return mobilenet_v3 model with frozen features layers"""
    model_small = mobilenet_v3_small(weights=weights)
    model_small.classifier[-1] = nn.Linear(in_features=1024, out_features=5)

    # deactivate gradient computation the features layers
    for param in model_small.features.parameters():
        param.requires_grad = False

    return model_small


def get_vgg16(
    n_classes: int = 5, weights=VGG16_Weights.IMAGENET1K_V1
) -> torch.nn.Module:
    """Return VGG16 model with frozen features layers"""
    vgg_model = vgg16(weights=weights)
    vgg_model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=n_classes)

    for param in vgg_model.features.parameters():
        param.requires_grad = False

    return vgg_model
