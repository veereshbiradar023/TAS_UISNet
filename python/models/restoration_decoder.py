"""
TAS-UISNet: Task-Conditioned Restoration Decoder with Instance-Conditioned Semantic Modulation (ICSM)
and Frequency-Spatial Feature Integration (FSFI)
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpatialBlock(nn.Module):
    """
    Frequency-Spatial Feature Integration (FSFI) Block.
    Separates low-frequency color attenuation / backscatter from high-frequency
    structural contours using 2D Fast Fourier Transform (FFT).
    """
    def __init__(self, channels: int):
        super(FrequencySpatialBlock, self).__init__()
        # Spatial branch: captures localized high-frequency edges and textures
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        # Frequency branch: captures global illumination and color balance
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1)
        )
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        feat_spatial = self.spatial_conv(x)

        # Frequency branch via 2D real FFT
        _, _, h, w = x.shape
        x_fft = torch.fft.rfft2(x, norm='backward')
        x_fft_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)
        freq_out = self.freq_conv(x_fft_cat)
        real_part, imag_part = torch.chunk(freq_out, 2, dim=1)
        x_fft_complex = torch.complex(real_part, imag_part)
        feat_freq = torch.fft.irfft2(x_fft_complex, s=(h, w), norm='backward')

        # Combine spatial and frequency representations
        combined = torch.cat([feat_spatial, feat_freq], dim=1)
        return x + self.fusion(combined)


class InstanceConditionedSemanticModulation(nn.Module):
    """
    Instance-Conditioned Semantic Modulation (ICSM).
    Computes scale (gamma) and shift (beta) modulation parameters from the
    instance semantic prior to selectively modulate restoration features:
    F_mod = gamma(S) * F + beta(S)
    """
    def __init__(self, feat_channels: int, semantic_channels: int = 1):
        super(InstanceConditionedSemanticModulation, self).__init__()
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(semantic_channels, feat_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, feat_channels, kernel_size=1)
        )
        self.mlp_beta = nn.Sequential(
            nn.Conv2d(semantic_channels, feat_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, feat_channels, kernel_size=1)
        )

    def forward(self, feat: torch.Tensor, semantic_prior: torch.Tensor) -> torch.Tensor:
        if semantic_prior.shape[-2:] != feat.shape[-2:]:
            semantic_prior = F.interpolate(semantic_prior, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        gamma = self.mlp_gamma(semantic_prior)
        beta = self.mlp_beta(semantic_prior)
        return (1.0 + gamma) * feat + beta


class TaskConditionedRestorationDecoder(nn.Module):
    """
    Task-Conditioned Restoration Decoder.
    Synthesizes multi-scale encoder features (P2-P5) conditioned on instance semantic feedback
    to produce enhanced RGB image and pixel-level restoration aleatoric uncertainty.
    """
    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super(TaskConditionedRestorationDecoder, self).__init__()
        
        # Semantic Modulation modules for each scale
        self.icsm_p5 = InstanceConditionedSemanticModulation(in_channels)
        self.icsm_p4 = InstanceConditionedSemanticModulation(in_channels)
        self.icsm_p3 = InstanceConditionedSemanticModulation(in_channels)
        self.icsm_p2 = InstanceConditionedSemanticModulation(in_channels)

        # Frequency-Spatial Integration blocks
        self.fsfi_p5 = FrequencySpatialBlock(in_channels)
        self.fsfi_p4 = FrequencySpatialBlock(in_channels)
        self.fsfi_p3 = FrequencySpatialBlock(in_channels)
        self.fsfi_p2 = FrequencySpatialBlock(in_channels)

        # Decoder Upsampling Convolutions
        self.up_p5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.fuse_p4 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.fuse_p3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.fuse_p2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

        # Final full-resolution reconstruction head (1/4 -> 1/1)
        self.recon_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),  # 1/2
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 1/1
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU()
        )

        # Residual image prediction: I_enhanced = clamp(I_input + delta_I, 0, 1)
        self.pred_rgb = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

        # Restoration aleatoric uncertainty prediction (log-variance log(sigma_r^2))
        self.pred_logvar = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, features: dict, semantic_prior: torch.Tensor, original_image: torch.Tensor):
        p2, p3, p4, p5 = features['P2'], features['P3'], features['P4'], features['P5']

        # Modulate and enhance coarse-to-fine
        p5_mod = self.fsfi_p5(self.icsm_p5(p5, semantic_prior))
        p5_up = F.interpolate(p5_mod, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        
        p4_cat = torch.cat([self.icsm_p4(p4, semantic_prior), p5_up], dim=1)
        p4_mod = self.fsfi_p4(self.fuse_p4(p4_cat))
        p4_up = F.interpolate(p4_mod, size=p3.shape[-2:], mode='bilinear', align_corners=False)

        p3_cat = torch.cat([self.icsm_p3(p3, semantic_prior), p4_up], dim=1)
        p3_mod = self.fsfi_p3(self.fuse_p3(p3_cat))
        p3_up = F.interpolate(p3_mod, size=p2.shape[-2:], mode='bilinear', align_corners=False)

        p2_cat = torch.cat([self.icsm_p2(p2, semantic_prior), p3_up], dim=1)
        p2_mod = self.fsfi_p2(self.fuse_p2(p2_cat))

        # Reconstruct to full resolution
        recon_feat = self.recon_head(p2_mod)
        if recon_feat.shape[-2:] != original_image.shape[-2:]:
            recon_feat = F.interpolate(recon_feat, size=original_image.shape[-2:], mode='bilinear', align_corners=False)

        delta_rgb = torch.tanh(self.pred_rgb(recon_feat)) * 0.5  # residual bounded offset
        enhanced_image = torch.clamp(original_image + delta_rgb, 0.0, 1.0)
        log_var = self.pred_logvar(recon_feat)  # log(sigma^2)

        return enhanced_image, log_var, recon_feat

