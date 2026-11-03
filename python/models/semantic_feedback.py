"""
TAS-UISNet: Uncertainty-Guided Semantic Feedback and Cross-Task Interaction Module
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFeedbackModule(nn.Module):
    """
    Bidirectional Uncertainty-Guided Semantic Feedback Module.
    1. Seg -> Rest: Semantic prior filtered by uncertainty gate modulates restoration features.
    2. Rest -> Seg: Enhanced textural features filtered by restoration confidence refine segmentation representations.
    """
    def __init__(self, channels: int = 256):
        super(SemanticFeedbackModule, self).__init__()
        
        # Seg -> Rest feedback projection
        self.seg_to_rest_proj = nn.Sequential(
            nn.Conv2d(1, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Rest -> Seg feature fusion projection
        self.rest_to_seg_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        # Cross-task adaptive weight parameter
        self.gamma_cross = nn.Parameter(torch.zeros(1))

    def forward(self, 
                coarse_semantic_prior: torch.Tensor, 
                rest_feat: torch.Tensor, 
                seg_feat: torch.Tensor, 
                collaboration_gate: torch.Tensor):
        """
        coarse_semantic_prior: (B, 1, H, W)
        rest_feat: (B, C, H', W')
        seg_feat: (B, C, H', W')
        collaboration_gate: (B, 1, H', W')
        """
        # Ensure spatial matching
        if coarse_semantic_prior.shape[-2:] != rest_feat.shape[-2:]:
            coarse_prior_rescaled = F.interpolate(coarse_semantic_prior, size=rest_feat.shape[-2:], mode='bilinear', align_corners=False)
        else:
            coarse_prior_rescaled = coarse_semantic_prior

        if collaboration_gate.shape[-2:] != rest_feat.shape[-2:]:
            gate = F.interpolate(collaboration_gate, size=rest_feat.shape[-2:], mode='bilinear', align_corners=False)
        else:
            gate = collaboration_gate

        # 1. Gated Semantic Prior for Restoration
        # High uncertainty regions suppress noisy semantic priors
        gated_semantic_prior = self.seg_to_rest_proj(coarse_prior_rescaled) * gate

        # 2. Gated Restoration Feature injection into Segmentation
        rest_inj = self.rest_to_seg_proj(rest_feat) * gate
        refined_seg_feat = seg_feat + self.gamma_cross * rest_inj

        return gated_semantic_prior, refined_seg_feat

