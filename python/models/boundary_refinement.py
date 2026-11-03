"""
TAS-UISNet: Boundary-Preserving Instance Separation and Refinement Module
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefinementModule(nn.Module):
    """
    Boundary Refinement Module (BRM).
    Uses high-frequency edge operators (Sobel/Laplacian) and contour-sensitive
    convolutions to disentangle touching and overlapping marine instances.
    """
    def __init__(self, in_channels: int = 256):
        super(BoundaryRefinementModule, self).__init__()
        
        # Fixed Sobel filters for edge-gradient extraction
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Boundary feature extraction
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Boundary-guided spatial recalibration
        self.attn_gate = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Refined output projection
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def extract_gradient_maps(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Computes 2D gradients across channel average.
        """
        mean_feat = torch.mean(feat, dim=1, keepdim=True)
        gx = F.conv2d(mean_feat, self.sobel_x, padding=1)
        gy = F.conv2d(mean_feat, self.sobel_y, padding=1)
        return torch.cat([gx, gy], dim=1)

    def forward(self, feat: torch.Tensor, coarse_mask: torch.Tensor = None):
        """
        feat: feature map (e.g. P2 from encoder or RoI features)
        coarse_mask: optional initial mask prediction
        """
        grads = self.extract_gradient_maps(feat)
        cat_in = torch.cat([feat, grads], dim=1)
        
        # Predict boundary probability map B_pred in [0, 1]
        boundary_logits = self.boundary_conv(cat_in)
        boundary_map = torch.sigmoid(boundary_logits)

        # Modulate feature map with boundary awareness
        attn = self.attn_gate(torch.cat([feat, boundary_map], dim=1))
        refined_feat = self.refine_conv(feat * (1.0 + attn))

        return refined_feat, boundary_map, boundary_logits

