import torch
import torch.nn as nn
import torchvision.models as models
from ssd.modeling import registry

class ResNet50(nn.Module):
    """ResNet50 backbone for SSD using torchvision's ResNet50."""
    def __init__(self, cfg):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1   # Output: 256 channels; size remains 75×75 → 75×75
        self.layer2 = resnet.layer2   # Output: 512 channels; size becomes ~38×38
        self.layer3 = resnet.layer3   # Output: 1024 channels; size ~19×19
        self.layer4 = resnet.layer4   # Output: 2048 channels; size ~10×10

        # Updated extra layer to downsample from 10x10 to 5x5
        self.extra_layer = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = []
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        features.append(x)  # Feature map 1: size 38×38, 512 channels
        x = self.layer3(x)
        features.append(x)  # Feature map 2: size 19×19, 1024 channels
        x = self.layer4(x)
        features.append(x)  # Feature map 3: size 10×10, 2048 channels
        x = self.extra_layer(x)
        features.append(x)  # Extra feature map: size 5×5, 512 channels

        return tuple(features)

@registry.BACKBONES.register('resnet50')
def resnet50_backbone(cfg, pretrained=True):
    model = ResNet50(cfg)
    return model