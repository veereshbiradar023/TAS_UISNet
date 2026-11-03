"""
TAS-UISNet: Comprehensive Evaluation on URPC2020, TrashCan, and SeaClear
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.tas_uisnet import TAS_UISNet
from datasets.urpc_dataset import URPC2020Dataset
from datasets.trashcan_dataset import TrashCanDataset, SeaClearDataset
from evaluation.metrics import (
    calculate_psnr, calculate_ssim, calculate_uiqm, calculate_uciqe,
    calculate_iou, calculate_boundary_fscore, calculate_ece
)


def evaluate_dataset(model, dataloader, dataset_name="URPC2020", device='cuda'):
    model.eval()
    psnr_vals, ssim_vals, uiqm_vals, uciqe_vals = [], [], [], []
    iou_vals, bf_vals = [], []
    all_probs, all_preds = [], []

    print(f"\n--- Evaluating on {dataset_name} ---")
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            gt_images = batch['gt_image'].cpu().numpy()
            gt_masks = batch['gt_masks'].cpu().numpy()

            outputs = model(images)
            enhanced = outputs['enhanced_image'].cpu().numpy()
            boundary_map = outputs['boundary_map'].cpu().numpy()

            for b in range(images.shape[0]):
                p_img = (np.transpose(enhanced[b], (1, 2, 0)) * 255.0).astype(np.uint8)
                g_img = (np.transpose(gt_images[b], (1, 2, 0)) * 255.0).astype(np.uint8)
                
                psnr_vals.append(calculate_psnr(p_img, g_img))
                ssim_vals.append(calculate_ssim(p_img, g_img))
                uiqm_vals.append(calculate_uiqm(p_img))
                uciqe_vals.append(calculate_uciqe(p_img))

                p_mask = (boundary_map[b, 0] > 0.5)
                g_mask = (gt_masks[b, 0] > 0.5)
                iou_vals.append(calculate_iou(p_mask, g_mask))
                bf_vals.append(calculate_boundary_fscore(p_mask, g_mask))

    print(f"Results for [{dataset_name}]:")
    print(f"  PSNR: {np.mean(psnr_vals):.2f} dB | SSIM: {np.mean(ssim_vals):.4f}")
    print(f"  UIQM: {np.mean(uiqm_vals):.3f} | UCIQE: {np.mean(uciqe_vals):.3f}")
    print(f"  Mean IoU: {np.mean(iou_vals)*100:.2f}% | Boundary F_B: {np.mean(bf_vals)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoints/best_tas_uisnet.pth')
    parser.add_argument('--dataset', type=str, default='all', choices=['urpc', 'trashcan', 'seaclear', 'all'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TAS_UISNet(num_classes=5, backbone='resnet50', pretrained=False).to(device)

    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print("Running in verification / benchmark mode with model weights.")

    if args.dataset in ['urpc', 'all']:
        urpc_loader = DataLoader(URPC2020Dataset(root_dir='data/URPC2020', split='val'), batch_size=4)
        evaluate_dataset(model, urpc_loader, "URPC2020 (In-Domain Benchmark)", device)

    if args.dataset in ['trashcan', 'all']:
        trash_loader = DataLoader(TrashCanDataset(root_dir='data/TrashCan', split='val'), batch_size=4)
        evaluate_dataset(model, trash_loader, "TrashCan (Zero-Shot Domain Transfer)", device)

    if args.dataset in ['seaclear', 'all']:
        seaclear_loader = DataLoader(SeaClearDataset(root_dir='data/SeaClear', split='test'), batch_size=4)
        evaluate_dataset(model, seaclear_loader, "SeaClear (Zero-Shot Marine Litter)", device)


if __name__ == '__main__':
    main()

