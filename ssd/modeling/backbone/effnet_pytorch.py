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
        self.layer0 = nn.Sequential(*features[:3])   # Expected output channels: 32; ~75x75
        self.layer1 = nn.Sequential(*features[3:4])    # Expected output channels: 48; ~38x38
        self.layer2 = nn.Sequential(*features[4:5])    # Expected output channels: 96; ~19x19
        self.layer3 = nn.Sequential(*features[5:7])    # Expected output channels: 232; ~10x10
        
        # Extra layers to generate additional feature maps using separate conv layers
        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(232, 512, kernel_size=3, stride=2, padding=1),  # Expected output: 512 channels; ~5x5 -> 3x3
            nn.ReLU(inplace=True)
        )
        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # Expected output: 256 channels; ~3x3 -> 1x1
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = []
        
        x = self.layer0(x)
        # print(f"Shape after layer0: {x.shape}")  # For debugging
        features.append(x) 
        
        x = self.layer1(x)
        # print(f"Shape after layer1: {x.shape}")  
        features.append(x)  

        x = self.layer2(x)
        # print(f"Shape after layer2: {x.shape}")  
        features.append(x)  

        x = self.layer3(x)
        # print(f"Shape after layer3: {x.shape}")  
        features.append(x)  

        x1 = self.extra_conv1(x)
        # print(f"Shape after extra_conv1: {x1.shape}")  
        features.append(x1)

        x2 = self.extra_conv2(x1)
        # print(f"Shape after extra_conv2: {x2.shape}")  
        features.append(x2)
        
        return tuple(features)

@registry.BACKBONES.register('efficientnet_b3')
def efficientnet_b3_backbone(cfg, pretrained=True):
    model = EfficientNetB3(cfg)
    return model