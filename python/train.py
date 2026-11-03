"""
TAS-UISNet: Main Training Entrypoint
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.urpc_dataset import URPC2020Dataset
from training.trainer import TAS_UISNet_Trainer


def main():
    parser = argparse.ArgumentParser(description="Train TAS-UISNet on URPC2020")
    parser.add_argument('--config', type=str, default='python/configs/tas_uisnet_config.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    config = {
        'num_classes': 5,
        'backbone': 'resnet50',
        'pretrained': True,
        'feat_dim': 256,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'use_amp': True,
        'checkpoint_dir': 'checkpoints',
        'lambda_r': 1.0,
        'lambda_c': 1.0,
        'lambda_b': 1.0,
        'lambda_m': 1.0,
        'lambda_e': 0.5,
        'lambda_u': 0.2,
        'lambda_f': 0.3
    }

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)

    print("=== Initializing TAS-UISNet Training Session ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Backbone: {config['backbone']}, Batch size: {config['batch_size']}, Epochs: {config['epochs']}")

    train_dataset = URPC2020Dataset(root_dir='data/URPC2020', split='train')
    val_dataset = URPC2020Dataset(root_dir='data/URPC2020', split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    trainer = TAS_UISNet_Trainer(config)

    best_psnr = 0.0
    for epoch in range(1, config['epochs'] + 1):
        loss = trainer.train_epoch(train_loader, epoch)
        if epoch % 5 == 0 or epoch == config['epochs']:
            psnr, ssim = trainer.validate(val_loader)
            is_best = psnr > best_psnr
            if is_best:
                best_psnr = psnr
            trainer.save_checkpoint(epoch, best=is_best)

    print("Training successfully completed!")


if __name__ == '__main__':
    main()

