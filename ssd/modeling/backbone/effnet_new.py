import torch
import torch.nn as nn
import torchvision.models as models
from ssd.modeling import registry

class EfficientNetB3(nn.Module):
    """EfficientNet-B3 backbone for SSD with six feature maps."""
    def __init__(self, cfg):
        super().__init__()
        effnet = models.efficientnet_b3(pretrained=True)
        features = list(effnet.features.children())
        
        # Split the features into sections
        self.layer0 = nn.Sequential(*features[:3])   # Output: ~38x38 (stride 8)
        self.layer1 = nn.Sequential(*features[3:4])  # Output: ~19x19 (stride 16)
        self.layer2 = nn.Sequential(*features[4:5])  # Output: ~10x10 (stride 32)
        self.layer3 = nn.Sequential(*features[5:7])  # Output: ~5x5 (stride 64)
        
        # Extra layers to generate additional feature maps
        self.extra_layers = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=3, stride=2, padding=1),  # Output: 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),   # Output: 1x1
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = []
        x = self.layer0(x)
        features.append(x)  # Feature map 1: 38x38
        x = self.layer1(x)
        features.append(x)  # Feature map 2: 19x19
        x = self.layer2(x)
        features.append(x)  # Feature map 3: 10x10
        x = self.layer3(x)
        features.append(x)  # Feature map 4: 5x5
        x = self.extra_layers(x)
        features.extend([x[:, :512], x[:, 512:]])  # Split into two maps: 3x3 and 1x1
        
        return tuple(features)

@registry.BACKBONES.register('efficientnet_b3')
def efficientnet_b3_backbone(cfg, pretrained=True):
    model = EfficientNetB3(cfg)
    return model