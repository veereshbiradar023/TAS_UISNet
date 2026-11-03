"""
TAS-UISNet: Multi-Task Loss Functions and Calibration Objectives
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Robust L1-variant for image reconstruction."""
    def __init__(self, eps: float = 1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity (SSIM) Loss."""
    def __init__(self, window_size: int = 11, channels: int = 3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channels = channels
        # Create Gaussian window
        kernel = self._create_window(window_size, channels)
        self.register_buffer('window', kernel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        padd = self.window_size // 2
        mu1 = F.conv2d(img1, self.window, padding=padd, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=padd, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padd, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padd, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=padd, groups=self.channels) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - torch.clamp(ssim_map.mean(), 0.0, 1.0)


class ColorAndFrequencyLoss(nn.Module):
    """
    Combined Color Consistency and Frequency Domain Loss.
    Penalizes spectral magnitude discrepancies and color balance shifts.
    """
    def __init__(self):
        super(ColorAndFrequencyLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Color loss via cosine similarity in RGB color vectors
        pred_norm = F.normalize(pred + 1e-7, p=2, dim=1)
        target_norm = F.normalize(target + 1e-7, p=2, dim=1)
        color_loss = 1.0 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))

        # Frequency domain loss via FFT
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        freq_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

        return color_loss + 0.1 * freq_loss


class HeteroscedasticUncertaintyLoss(nn.Module):
    """
    Heteroscedastic Aleatoric Uncertainty Loss:
    L_unc = 0.5 * exp(-s) * ||pred - target||^2 + 0.5 * s
    where s = log(sigma^2)
    """
    def __init__(self):
        super(HeteroscedasticUncertaintyLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # Clamp log_var to avoid numerical instability
        log_var = torch.clamp(log_var, min=-6.0, max=4.0)
        precision = torch.exp(-log_var)
        diff_sq = (pred - target) ** 2
        loss = 0.5 * precision * diff_sq + 0.5 * log_var
        return torch.mean(loss)


class BoundaryLoss(nn.Module):
    """
    Boundary Focal + Dice Loss for crisp instance boundary delineation.
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> torch.Tensor:
        if pred_boundary.shape[-2:] != gt_boundary.shape[-2:]:
            pred_boundary = F.interpolate(pred_boundary, size=gt_boundary.shape[-2:], mode='bilinear', align_corners=False)
        
        bce = F.binary_cross_entropy(pred_boundary, gt_boundary)
        # Dice loss
        intersection = torch.sum(pred_boundary * gt_boundary)
        dice = 1.0 - (2.0 * intersection + 1.0) / (torch.sum(pred_boundary) + torch.sum(gt_boundary) + 1.0)
        return bce + dice


class TAS_UISNet_TotalLoss(nn.Module):
    """
    Unified Multi-Task Objective Function for TAS-UISNet.
    L_total = lambda_r * L_restoration
            + lambda_c * L_classification
            + lambda_b * L_box
            + lambda_m * L_mask
            + lambda_e * L_boundary
            + lambda_u * L_uncertainty
            + lambda_f * L_feature_collab
    """
    def __init__(self,
                 lambda_r: float = 1.0,
                 lambda_c: float = 1.0,
                 lambda_b: float = 1.0,
                 lambda_m: float = 1.0,
                 lambda_e: float = 0.5,
                 lambda_u: float = 0.2,
                 lambda_f: float = 0.3):
        super(TAS_UISNet_TotalLoss, self).__init__()
        self.lambda_r = lambda_r
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.lambda_m = lambda_m
        self.lambda_e = lambda_e
        self.lambda_u = lambda_u
        self.lambda_f = lambda_f

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.color_freq = ColorAndFrequencyLoss()
        self.heteroscedastic = HeteroscedasticUncertaintyLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, outputs: dict, targets: dict) -> dict:
        # 1. Restoration Loss: Charbonnier + SSIM + Color/Frequency
        l_char = self.charbonnier(outputs['enhanced_image'], targets['gt_image'])
        l_ssim = self.ssim(outputs['enhanced_image'], targets['gt_image'])
        l_colfreq = self.color_freq(outputs['enhanced_image'], targets['gt_image'])
        l_restoration = l_char + 0.2 * l_ssim + 0.1 * l_colfreq

        # 2. Uncertainty Loss
        l_uncertainty = self.heteroscedastic(
            outputs['enhanced_image'], targets['gt_image'], outputs['log_var']
        )

        # 3. Boundary Loss
        l_boundary = self.boundary_loss(outputs['boundary_map'], targets['gt_boundary'])

        # 4. Classification & Box Regression Loss (Smooth L1 + Cross Entropy)
        if 'gt_labels' in targets and targets['gt_labels'] is not None:
            l_cls = F.cross_entropy(outputs['cls_logits'], targets['gt_labels'])
        else:
            l_cls = torch.tensor(0.0, device=outputs['enhanced_image'].device)

        if 'gt_boxes' in targets and targets['gt_boxes'] is not None:
            l_box = F.smooth_l1_loss(outputs['bbox_deltas'], targets['gt_boxes'])
        else:
            l_box = torch.tensor(0.0, device=outputs['enhanced_image'].device)

        # 5. Mask Segmentation Loss
        if 'gt_masks' in targets and targets['gt_masks'] is not None:
            mask_pred = outputs['mask_logits']
            if mask_pred.shape[-2:] != targets['gt_masks'].shape[-2:]:
                mask_pred = F.interpolate(mask_pred, size=targets['gt_masks'].shape[-2:], mode='bilinear', align_corners=False)
            l_mask = F.binary_cross_entropy_with_logits(mask_pred, targets['gt_masks'])
        else:
            l_mask = torch.tensor(0.0, device=outputs['enhanced_image'].device)

        # 6. Feature Collaboration Consistency
        coarse_prior = outputs['coarse_semantic_prior']
        gt_semantic = targets.get('gt_semantic_mask', targets.get('gt_boundary'))
        if coarse_prior.shape[-2:] != gt_semantic.shape[-2:]:
            coarse_prior = F.interpolate(coarse_prior, size=gt_semantic.shape[-2:], mode='bilinear', align_corners=False)
        l_collab = F.mse_loss(coarse_prior, gt_semantic)

        # Total Weighted Sum
        total_loss = (
            self.lambda_r * l_restoration +
            self.lambda_u * l_uncertainty +
            self.lambda_e * l_boundary +
            self.lambda_c * l_cls +
            self.lambda_b * l_box +
            self.lambda_m * l_mask +
            self.lambda_f * l_collab
        )

        return {
            'loss_total': total_loss,
            'loss_restoration': l_restoration,
            'loss_uncertainty': l_uncertainty,
            'loss_boundary': l_boundary,
            'loss_cls': l_cls,
            'loss_box': l_box,
            'loss_mask': l_mask,
            'loss_collab': l_collab
        }

