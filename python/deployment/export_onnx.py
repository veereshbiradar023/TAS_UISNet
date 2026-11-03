"""
TAS-UISNet: ONNX and TensorRT Export Pipeline for Edge Deployment on NVIDIA Jetson Orin
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import os
import argparse
import torch
import time
import numpy as np

from models.tas_uisnet import TAS_UISNet


class TAS_UISNet_ExportWrapper(torch.nn.Module):
    """
    Export wrapper simplifying the return tuple for ONNX / TensorRT runtime execution.
    """
    def __init__(self, model):
        super(TAS_UISNet_ExportWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return (
            out['enhanced_image'],
            out['restoration_uncertainty'],
            out['boundary_map'],
            out['cls_logits'],
            out['bbox_deltas'],
            out['mask_logits']
        )


def export_to_onnx(weights_path: str, output_path: str, height: int = 480, width: int = 640):
    print(f"Loading weights from {weights_path}...")
    model = TAS_UISNet(num_classes=5, backbone='resnet50', pretrained=False)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    export_wrapper = TAS_UISNet_ExportWrapper(model)
    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Exporting ONNX model to {output_path}...")

    torch.onnx.export(
        export_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=[
            'enhanced_image',
            'restoration_uncertainty',
            'boundary_map',
            'cls_logits',
            'bbox_deltas',
            'mask_logits'
        ],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'enhanced_image': {0: 'batch_size'},
            'restoration_uncertainty': {0: 'batch_size'},
            'boundary_map': {0: 'batch_size'}
        }
    )
    print("ONNX export successfully finished!")


def benchmark_edge_latency(device_name='Jetson Orin (Simulated / Local)', iters=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TAS_UISNet(num_classes=5, backbone='resnet50', pretrained=False).to(device)
    model.eval()
    dummy = torch.randn(1, 3, 480, 640, device=device)

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    total_time = time.time() - start
    avg_latency_ms = (total_time / iters) * 1000.0
    fps = 1000.0 / avg_latency_ms
    print(f"[{device_name}] Average Latency: {avg_latency_ms:.2f} ms | Throughput: {fps:.1f} FPS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoints/best_tas_uisnet.pth')
    parser.add_argument('--output', type=str, default='python/deployment/tas_uisnet_orin.onnx')
    args = parser.parse_args()
    
    export_to_onnx(args.weights, args.output)
    benchmark_edge_latency()

