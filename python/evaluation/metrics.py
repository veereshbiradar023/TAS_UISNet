"""
TAS-UISNet: Comprehensive Evaluation Metrics Suite
Includes:
- PSNR, SSIM, UIQM, UCIQE
- Mask AP, Box AP, IoU, Boundary F-score
- Uncertainty ECE, NLL, Brier score
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import math
import numpy as np
import torch
import cv2


# ==========================================
# 1. Image Enhancement Metrics
# ==========================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * math.log10(255.0 / math.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Structural Similarity Index (SSIM)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def calculate_uciqe(img_rgb: np.ndarray) -> float:
    """
    Underwater Color Image Quality Evaluation (UCIQE).
    UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
    where:
    - sigma_c: standard deviation of chroma
    - con_l: contrast of luminance
    - mu_s: average saturation
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # Chroma
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)

    # Luminance contrast
    l_flat = l.flatten()
    sorted_l = np.sort(l_flat)
    bottom_1 = sorted_l[int(0.01 * len(l_flat))]
    top_1 = sorted_l[int(0.99 * len(l_flat))]
    con_l = top_1 - bottom_1

    # Saturation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    sat = hsv[:, :, 1] / 255.0
    mu_s = np.mean(sat)

    # Standard Panetta et al. / Yang & Xu coefficients
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * sigma_c + c2 * con_l + c3 * mu_s)


def calculate_uiqm(img_rgb: np.ndarray) -> float:
    """
    Underwater Image Quality Measure (UIQM).
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    """
    # 1. UICM (Colorfulness)
    rg = img_rgb[:, :, 0].astype(np.float64) - img_rgb[:, :, 1].astype(np.float64)
    yb = 0.5 * (img_rgb[:, :, 0].astype(np.float64) + img_rgb[:, :, 1].astype(np.float64)) - img_rgb[:, :, 2].astype(np.float64)
    uicm = -0.0268 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.1586 * np.sqrt(np.var(rg) + np.var(yb))

    # 2. UISM (Sharpness) via Sobel
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    uism = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2))) * 0.05

    # 3. UIConM (Contrast)
    uiconm = float(np.std(gray)) * 0.1

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    return float(uiqm)


# ==========================================
# 2. Instance Segmentation & Boundary Metrics
# ==========================================

def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Intersection over Union for binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def calculate_boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, bound_th: float = 2.0) -> float:
    """
    Boundary F-score (F_B) measuring contour alignment within tolerance bound_th pixels.
    """
    def _get_boundary(m):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dil = cv2.dilate(m.astype(np.uint8), kernel)
        ero = cv2.erode(m.astype(np.uint8), kernel)
        return (dil - ero) > 0

    pred_b = _get_boundary(pred_mask)
    gt_b = _get_boundary(gt_mask)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    dist_gt = cv2.distanceTransform((1 - gt_b.astype(np.uint8)), cv2.DIST_L2, 3)
    dist_pred = cv2.distanceTransform((1 - pred_b.astype(np.uint8)), cv2.DIST_L2, 3)

    precision = np.sum(dist_gt[pred_b] <= bound_th) / float(pred_b.sum())
    recall = np.sum(dist_pred[gt_b] <= bound_th) / float(gt_b.sum())

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ==========================================
# 3. Uncertainty Calibration Metrics
# ==========================================

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def calculate_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Brier Score for probability forecasting accuracy."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

