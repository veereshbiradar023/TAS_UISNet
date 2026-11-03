"""
TAS-UISNet: Training Pipeline with Mixed-Precision (AMP) and Cosine Annealing
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.tas_uisnet import TAS_UISNet
from losses.total_loss import TAS_UISNet_TotalLoss
from evaluation.metrics import calculate_psnr, calculate_ssim, calculate_uiqm


class TAS_UISNet_Trainer:
    """
    Standard Trainer supporting RTX 4090 / A100 training with mixed precision.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Instantiate Model
        self.model = TAS_UISNet(
            num_classes=config.get('num_classes', 5),
            backbone=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            feat_dim=config.get('feat_dim', 256),
            iterative_inference=config.get('iterative_inference', False)
        ).to(self.device)

        # Loss and Optimizer
        self.criterion = TAS_UISNet_TotalLoss(
            lambda_r=config.get('lambda_r', 1.0),
            lambda_c=config.get('lambda_c', 1.0),
            lambda_b=config.get('lambda_b', 1.0),
            lambda_m=config.get('lambda_m', 1.0),
            lambda_e=config.get('lambda_e', 0.5),
            lambda_u=config.get('lambda_u', 0.2),
            lambda_f=config.get('lambda_f', 0.3)
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        epochs = config.get('epochs', 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.scaler = GradScaler(enabled=config.get('use_amp', True))
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            targets = {
                'gt_image': batch['gt_image'].to(self.device),
                'gt_masks': batch['gt_masks'].to(self.device),
                'gt_boundary': batch['gt_boundary'].to(self.device),
                'gt_semantic_mask': batch['gt_semantic_mask'].to(self.device),
                'gt_labels': batch.get('gt_labels', None),
                'gt_boxes': batch.get('gt_boxes', None)
            }
            if targets['gt_labels'] is not None:
                targets['gt_labels'] = targets['gt_labels'].to(self.device)
            if targets['gt_boxes'] is not None:
                targets['gt_boxes'] = targets['gt_boxes'].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.get('use_amp', True)):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['loss_total']

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch:02d}][{batch_idx:03d}/{len(dataloader):03d}] "
                      f"Loss: {loss.item():.4f} (Rest: {loss_dict['loss_restoration'].item():.3f}, "
                      f"Unc: {loss_dict['loss_uncertainty'].item():.3f}, "
                      f"Bound: {loss_dict['loss_boundary'].item():.3f})")

        self.scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, dataloader: DataLoader):
        self.model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                gt_images = batch['gt_image'].cpu().numpy()
                outputs = self.model(images)
                enhanced = outputs['enhanced_image'].cpu().numpy()

                for b in range(images.shape[0]):
                    p_img = (np.transpose(enhanced[b], (1, 2, 0)) * 255.0).astype(np.uint8)
                    g_img = (np.transpose(gt_images[b], (1, 2, 0)) * 255.0).astype(np.uint8)
                    psnr_list.append(calculate_psnr(p_img, g_img))
                    ssim_list.append(calculate_ssim(p_img, g_img))

        mean_psnr = np.mean(psnr_list) if psnr_list else 0.0
        mean_ssim = np.mean(ssim_list) if ssim_list else 0.0
        print(f"Validation PSNR: {mean_psnr:.2f} dB, SSIM: {mean_ssim:.4f}")
        return mean_psnr, mean_ssim

    def save_checkpoint(self, epoch: int, best: bool = False):
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, ckpt_path)
        if best:
            best_path = os.path.join(self.checkpoint_dir, "best_tas_uisnet.pth")
            torch.save(self.model.state_dict(), best_path)
            print(f"Saved new best model checkpoint to {best_path}")

