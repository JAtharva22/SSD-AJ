import torch
import torch.nn as nn
import torchvision.models as models
from ssd.modeling import registry

class InceptionV3(nn.Module):
    """Inception V3 backbone for SSD that outputs 6 feature maps."""
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        inception = models.inception_v3(pretrained=pretrained)
        # Manually assign inception layers for easier feature extraction:
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d   # Feature 1 will be from here
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e   # Feature 2 from here
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c   # Feature 3 from here

        # Extra layers to create additional (coarser) feature maps
        self.extra1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.extra3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = []
        # Base network forward pass
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        features.append(x)  # Feature 1: from Mixed_5d
        
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        features.append(x)  # Feature 2: from Mixed_6e
        
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        features.append(x)  # Feature 3: from Mixed_7c
        
        # Extra layers for additional feature maps:
        x = self.extra1(x)
        features.append(x)  # Feature 4: extra layer 1 (512 channels)
        
        x = self.extra2(x)
        features.append(x)  # Feature 5: extra layer 2 (256 channels)
        
        x = self.extra3(x)
        features.append(x)  # Feature 6: extra layer 3 (256 channels)
        
        return tuple(features)

@registry.BACKBONES.register('InceptionV3')
def inception_v3_ssd_backbone(cfg, pretrained=True):
    model = InceptionV3(cfg, pretrained=pretrained)
    return model
