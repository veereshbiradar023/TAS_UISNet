"""
TAS-UISNet: Shared Multi-Scale Hierarchical Encoder with Spatial-Channel Attention
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialChannelAttention(nn.Module):
    """
    Channel and Spatial Attention Module (CBAM-style) tailored for
    underwater turbidity suppression and salient feature recalibration.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SpatialChannelAttention, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention forward
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x_ca = x * channel_attn

        # Spatial attention forward
        avg_spatial = torch.mean(x_ca, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(spatial_cat))
        
        return x_ca * spatial_attn


class SharedMultiScaleEncoder(nn.Module):
    """
    Shared Multi-Scale Hierarchical Feature Encoder with FPN and Attentive Recalibration.
    Extracts multi-resolution representations (P2, P3, P4, P5) for both
    image restoration and instance segmentation.
    """
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, out_channels: int = 256):
        super(SharedMultiScaleEncoder, self).__init__()
        
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif backbone == 'resnet101':
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            resnet = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Stem
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # ResNet stages
        self.layer1 = resnet.layer1  # 256 channels (1/4 scale)
        self.layer2 = resnet.layer2  # 512 channels (1/8 scale)
        self.layer3 = resnet.layer3  # 1024 channels (1/16 scale)
        self.layer4 = resnet.layer4  # 2048 channels (1/32 scale)

        # Lateral 1x1 convolutions for Feature Pyramid Network (FPN)
        self.lat_layer4 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lat_layer1 = nn.Conv2d(256, out_channels, kernel_size=1)

        # Smooth 3x3 convolutions with attention
        self.smooth_p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SpatialChannelAttention(out_channels)
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SpatialChannelAttention(out_channels)
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SpatialChannelAttention(out_channels)
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SpatialChannelAttention(out_channels)
        )

    def forward(self, x: torch.Tensor):
        # Bottom-up pathway
        c1 = self.stem(x)
        c2 = self.layer1(c1)  # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32

        # Top-down pathway with lateral connections
        m5 = self.lat_layer4(c5)
        m4 = self.lat_layer3(c4) + F.interpolate(m5, size=c4.shape[-2:], mode='nearest')
        m3 = self.lat_layer2(c3) + F.interpolate(m4, size=c3.shape[-2:], mode='nearest')
        m2 = self.lat_layer1(c2) + F.interpolate(m3, size=c2.shape[-2:], mode='nearest')

        # Output feature maps (P2: 1/4, P3: 1/8, P4: 1/16, P5: 1/32)
        p5 = self.smooth_p5(m5)
        p4 = self.smooth_p4(m4)
        p3 = self.smooth_p3(m3)
        p2 = self.smooth_p2(m2)

        return {
            'P2': p2,
            'P3': p3,
            'P4': p4,
            'P5': p5
        }

