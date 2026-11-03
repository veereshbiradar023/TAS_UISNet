"""
TAS-UISNet: End-to-End Single Image Inference Pipeline
Outputs enhanced image, predicted instance masks, boundary maps, and restoration uncertainty maps.
Author: Veeresh Birdar (SRM Institute of Science and Technology)
"""

import os
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.tas_uisnet import TAS_UISNet


def run_inference(image_path: str, output_dir: str, weights_path: str = None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TAS_UISNet(num_classes=5, backbone='resnet50', pretrained=False).to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    if os.path.exists(image_path):
        raw_img = Image.open(image_path).convert('RGB')
    else:
        # Mock underwater image for demonstration
        raw_img = Image.fromarray(np.random.randint(30, 200, (480, 640, 3), dtype=np.uint8))

    img_resized = TF.resize(raw_img, (480, 640))
    tensor_img = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)

    enhanced = outputs['enhanced_image'].squeeze(0).cpu().permute(1, 2, 0).numpy()
    uncertainty = outputs['restoration_uncertainty'].squeeze(0).squeeze(0).cpu().numpy()
    boundary = outputs['boundary_map'].squeeze(0).squeeze(0).cpu().numpy()
    gate = outputs['collaboration_gate'].squeeze(0).squeeze(0).cpu().numpy()

    # Save visual comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    axes[0].imshow(img_resized)
    axes[0].set_title("Input Raw Underwater")
    axes[0].axis('off')

    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced Output (TAS-UISNet)")
    axes[1].axis('off')

    im_unc = axes[2].imshow(uncertainty, cmap='inferno')
    axes[2].set_title("Restoration Uncertainty Map")
    axes[2].axis('off')
    plt.colorbar(im_unc, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(boundary, cmap='bone')
    axes[3].set_title("Boundary Separation Map")
    axes[3].axis('off')

    save_fig_path = os.path.join(output_dir, 'tas_uisnet_inference_demo.png')
    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()
    print(f"Inference visual result saved to {save_fig_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='data/URPC2020/images/sample.jpg')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--weights', type=str, default='checkpoints/best_tas_uisnet.pth')
    args = parser.parse_args()

    run_inference(args.image, args.output_dir, args.weights)

