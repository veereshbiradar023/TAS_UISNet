"""
TAS-UISNet: Uncertainty Estimation Module (Aleatoric, Epistemic, and Calibration Gating)
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyModule(nn.Module):
    """
    Uncertainty Quantification and Calibration Module.
    Computes:
    1. Pixel-level restoration aleatoric variance (sigma_r^2)
    2. Segmentation uncertainty (predictive entropy and variance)
    3. Epistemic uncertainty through Monte Carlo Dropout sampling
    4. Calibrated uncertainty gating map for collaborative cross-task fusion
    """
    def __init__(self, feat_dim: int = 256, mc_samples: int = 5, temperature: float = 1.5):
        super(UncertaintyModule, self).__init__()
        self.mc_samples = mc_samples
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        # Spatial uncertainty projection
        self.proj_uncertainty = nn.Sequential(
            nn.Conv2d(feat_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Dropout layer for epistemic uncertainty estimation
        self.mc_dropout = nn.Dropout2d(p=0.3)

    def compute_pixel_aleatoric(self, log_var: torch.Tensor) -> torch.Tensor:
        """
        Computes standard deviation sigma = exp(0.5 * log_var).
        Normalized into [0, 1] range for visual and gating interpretation.
        """
        var = torch.exp(torch.clamp(log_var, min=-6.0, max=4.0))
        std = torch.sqrt(var)
        norm_std = torch.mean(std, dim=1, keepdim=True)
        norm_std = norm_std / (norm_std.max() + 1e-6)
        return norm_std

    def compute_epistemic_uncertainty(self, model_forward_func, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates epistemic uncertainty via Monte Carlo Dropout over T passes.
        """
        preds = []
        for _ in range(self.mc_samples):
            pred = model_forward_func(self.mc_dropout(x))
            preds.append(pred)
        preds = torch.stack(preds, dim=0)
        epistemic_var = torch.var(preds, dim=0).mean(dim=1, keepdim=True)
        return epistemic_var

    def compute_predictive_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes Shannon entropy H(p) = - sum p_c log(p_c) scaled by temperature.
        """
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1)
        probs = F.softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy

    def generate_uncertainty_gate(self, restoration_unc: torch.Tensor, segmentation_unc: torch.Tensor) -> torch.Tensor:
        """
        Generates uncertainty-guided collaboration gating map:
        Omega = exp( - (alpha * U_r + beta * U_s) )
        Where regions of high turbidity/noise are down-weighted to prevent error propagation.
        """
        if restoration_unc.shape[-2:] != segmentation_unc.shape[-2:]:
            segmentation_unc = F.interpolate(segmentation_unc, size=restoration_unc.shape[-2:], mode='bilinear', align_corners=False)
        
        combined_unc = 0.5 * restoration_unc + 0.5 * segmentation_unc
        # Gate factor ranges from near 0 (highly uncertain / noisy) to 1 (highly confident)
        gate = torch.exp(-2.0 * combined_unc)
        return gate

    def forward(self, feat: torch.Tensor, log_var: torch.Tensor, coarse_prior: torch.Tensor):
        restoration_unc = self.compute_pixel_aleatoric(log_var)
        feat_unc = self.proj_uncertainty(feat)
        if feat_unc.shape[-2:] != restoration_unc.shape[-2:]:
            feat_unc = F.interpolate(feat_unc, size=restoration_unc.shape[-2:], mode='bilinear', align_corners=False)

        gate = self.generate_uncertainty_gate(restoration_unc, feat_unc)
        return {
            'restoration_uncertainty': restoration_unc,
            'feature_uncertainty': feat_unc,
            'collaboration_gate': gate
        }

