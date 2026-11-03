"""
TAS-UISNet: Unified Architecture for Joint Underwater Image Enhancement and Instance Segmentation
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_encoder import SharedMultiScaleEncoder
from .restoration_decoder import TaskConditionedRestorationDecoder
from .instance_head import InstanceSegmentationHead
from .uncertainty_module import UncertaintyModule
from .semantic_feedback import SemanticFeedbackModule
from .boundary_refinement import BoundaryRefinementModule


class TAS_UISNet(nn.Module):
    """
    TAS-UISNet: Task-Aware Semantic Restoration and Uncertainty-Guided Feature Collaboration
    for Joint Underwater Image Enhancement and Instance Segmentation.
    """
    def __init__(self, 
                 num_classes: int = 5, 
                 backbone: str = 'resnet50', 
                 pretrained: bool = True,
                 feat_dim: int = 256,
                 iterative_inference: bool = False):
        super(TAS_UISNet, self).__init__()
        self.num_classes = num_classes
        self.iterative_inference = iterative_inference
        
        # 1. Shared Multi-Scale Hierarchical Encoder
        self.encoder = SharedMultiScaleEncoder(backbone=backbone, pretrained=pretrained, out_channels=feat_dim)

        # 2. Instance Segmentation Head (RPN + RoIAlign + Fast R-CNN + Mask)
        self.instance_head = InstanceSegmentationHead(in_channels=feat_dim, num_classes=num_classes)

        # 3. Task-Conditioned Restoration Decoder
        self.restoration_decoder = TaskConditionedRestorationDecoder(in_channels=feat_dim, out_channels=3)

        # 4. Uncertainty Estimation Module
        self.uncertainty_module = UncertaintyModule(feat_dim=feat_dim)

        # 5. Semantic Feedback Module
        self.semantic_feedback = SemanticFeedbackModule(channels=feat_dim)

        # 6. Boundary Refinement Module
        self.boundary_refinement = BoundaryRefinementModule(in_channels=feat_dim)

    def forward(self, x: torch.Tensor, rois: torch.Tensor = None):
        """
        Forward Pass:
        1. Extract multi-scale features P2-P5 via Shared Encoder.
        2. Generate coarse instance prior to decouple circular dependencies.
        3. Perform initial restoration pass and estimate restoration uncertainty.
        4. Synthesize uncertainty-calibrated collaboration gate.
        5. Modulate restoration features with gated semantic prior and feed back into segmentation.
        6. Apply boundary refinement to separate touching instances.
        7. Compute final enhanced image, class labels, boxes, masks, and boundary maps.
        """
        b, c, h, w = x.shape

        # Step 1: Multi-scale Feature Extraction
        features = self.encoder(x)
        p2 = features['P2']  # 1/4 resolution

        # Step 2: Coarse Semantic Prior Extraction (Non-circular)
        coarse_semantic_prior = self.instance_head.extract_coarse_prior(p2, (h, w))

        # Step 3: Task-Conditioned Restoration (Stage 1)
        enhanced_image, log_var, recon_feat = self.restoration_decoder(features, coarse_semantic_prior, x)

        # Step 4: Uncertainty Estimation & Collaboration Gating
        unc_dict = self.uncertainty_module(p2, log_var, coarse_semantic_prior)
        collab_gate = unc_dict['collaboration_gate']
        restoration_unc = unc_dict['restoration_uncertainty']

        # Step 5: Bidirectional Semantic Feedback
        # Downsample recon_feat to match P2 if necessary
        recon_feat_down = F.interpolate(recon_feat, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        gated_semantic_prior, refined_p2 = self.semantic_feedback(
            coarse_semantic_prior, recon_feat_down, p2, collab_gate
        )

        # Optional Iterative Inference step for restoration refinement
        if self.iterative_inference:
            enhanced_image, log_var, recon_feat = self.restoration_decoder(features, gated_semantic_prior, x)
            restoration_unc = self.uncertainty_module.compute_pixel_aleatoric(log_var)

        # Step 6: Boundary Refinement
        refined_feat, boundary_map, boundary_logits = self.boundary_refinement(refined_p2)

        # Step 7: Instance Segmentation Prediction
        inst_dict = self.instance_head(refined_feat, rois)

        return {
            'enhanced_image': enhanced_image,
            'log_var': log_var,
            'restoration_uncertainty': restoration_unc,
            'coarse_semantic_prior': coarse_semantic_prior,
            'collaboration_gate': collab_gate,
            'boundary_map': boundary_map,
            'boundary_logits': boundary_logits,
            'cls_logits': inst_dict['cls_logits'],
            'bbox_deltas': inst_dict['bbox_deltas'],
            'mask_logits': inst_dict['mask_logits'],
            'mask_uncertainty': inst_dict['mask_uncertainty'],
            'rois': inst_dict['rois']
        }

