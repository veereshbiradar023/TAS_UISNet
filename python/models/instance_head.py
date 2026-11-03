"""
TAS-UISNet: Instance Segmentation Head with Coarse-to-Fine Semantic Prior Extraction
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, nms


class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network (RPN) generating candidate bounding boxes.
    """
    def __init__(self, in_channels: int = 256, num_anchors: int = 3):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feat = F.relu(self.conv(x))
        logits = self.cls_logits(feat)
        bbox_deltas = self.bbox_pred(feat)
        return logits, bbox_deltas


class CoarseSemanticPriorExtractor(nn.Module):
    """
    Generates non-circular coarse semantic instance density maps from multi-scale features
    prior to full instance mask refinement. This breaks circular dependencies!
    """
    def __init__(self, in_channels: int = 256):
        super(CoarseSemanticPriorExtractor, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, p2: torch.Tensor, out_size: tuple) -> torch.Tensor:
        coarse_map = self.head(p2)
        if coarse_map.shape[-2:] != out_size:
            coarse_map = F.interpolate(coarse_map, size=out_size, mode='bilinear', align_corners=False)
        return coarse_map


class InstanceSegmentationHead(nn.Module):
    """
    Instance Segmentation Head containing RoI Feature Extractor, Box/Class Head,
    and FCN Mask Head.
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 5, roi_size: int = 14):
        super(InstanceSegmentationHead, self).__init__()
        self.num_classes = num_classes
        self.roi_size = roi_size
        
        # RoI align layer
        self.roi_align = RoIAlign(output_size=(roi_size, roi_size), spatial_scale=0.25, sampling_ratio=2)

        # Fast R-CNN Box / Class branch
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * roi_size * roi_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # Mask prediction branch (FCN)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Instance mask uncertainty projection
        self.mask_uncertainty_head = nn.Conv2d(256, 1, kernel_size=1)

        # Coarse semantic extractor to break circular feedback
        self.coarse_prior = CoarseSemanticPriorExtractor(in_channels)

    def extract_coarse_prior(self, p2: torch.Tensor, out_size: tuple) -> torch.Tensor:
        return self.coarse_prior(p2, out_size)

    def forward(self, feat_p2: torch.Tensor, rois: torch.Tensor = None):
        """
        rois: Tensor of shape (N, 5) with [batch_idx, x1, y1, x2, y2]
        If rois is None (e.g. testing or training dummy pass), auto-generate grid proposals.
        """
        b, c, h, w = feat_p2.shape
        if rois is None or rois.shape[0] == 0:
            # Generate standard full-image / grid candidate RoIs for inference/initialization
            rois = []
            for b_idx in range(b):
                rois.append(torch.tensor([[b_idx, 0.0, 0.0, float(w * 4), float(h * 4)]], device=feat_p2.device))
            rois = torch.cat(rois, dim=0)

        roi_features = self.roi_align(feat_p2, rois)
        
        # Classification and bounding box
        box_fc = self.box_head(roi_features)
        cls_logits = self.cls_score(box_fc)
        bbox_deltas = self.bbox_pred(box_fc)

        # Mask prediction
        mask_feat = roi_features
        for layer in list(self.mask_head.children())[:-1]:
            mask_feat = layer(mask_feat)
        mask_logits = self.mask_head[-1](mask_feat)
        
        # RoI-level mask uncertainty
        mask_uncertainty = torch.sigmoid(self.mask_uncertainty_head(mask_feat))

        return {
            'cls_logits': cls_logits,
            'bbox_deltas': bbox_deltas,
            'mask_logits': mask_logits,
            'mask_uncertainty': mask_uncertainty,
            'rois': rois
        }

