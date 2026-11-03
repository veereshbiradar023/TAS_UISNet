"""
TAS-UISNet: URPC2020 Dataset Loader with Boundary Map Generation
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

from .transforms import UnderwaterTransform


class URPC2020Dataset(Dataset):
    """
    Dataset loader for URPC2020 underwater benchmark.
    Classes:
    0: background
    1: echinus (sea urchin)
    2: holothurian (sea cucumber)
    3: scallop
    4: starfish
    """
    CLASSES = ('background', 'echinus', 'holothurian', 'scallop', 'starfish')

    def __init__(self, root_dir: str, split: str = 'train', transform=None, img_size=(480, 640)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else UnderwaterTransform(img_size=img_size, is_train=(split == 'train'))
        
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.ann_file = os.path.join(root_dir, 'annotations', f'{split}.json')
        
        self.samples = []
        if os.path.exists(self.ann_file):
            with open(self.ann_file, 'r') as f:
                self.coco_data = json.load(f)
            self.samples = self.coco_data.get('images', [])
        elif os.path.exists(self.image_dir):
            valid_exts = ('.jpg', '.png', '.jpeg')
            self.samples = [{'file_name': f, 'id': idx} for idx, f in enumerate(sorted(os.listdir(self.image_dir))) if f.lower().endswith(valid_exts)]
        else:
            # Synthetic / demonstration fallback placeholder
            self.samples = [{'file_name': f'urpc_{i:04d}.jpg', 'id': i} for i in range(50)]

    def __len__(self):
        return len(self.samples)

    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extracts 1-pixel crisp boundary from binary/instance mask using morphological gradient."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated - eroded
        return (boundary > 0).astype(np.float32)

    def __getitem__(self, idx: int):
        info = self.samples[idx]
        file_name = info['file_name']
        img_path = os.path.join(self.image_dir, file_name)

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            # Mock placeholder image for initialization and verification
            img = Image.fromarray(np.random.randint(20, 200, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8))

        # Check for paired reference / ground-truth clean image if available
        gt_path = os.path.join(self.root_dir, 'references', file_name)
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB')
        else:
            gt_img = img.copy()

        # Check for mask file or synthesize mock mask
        mask_path = os.path.join(self.root_dir, 'masks', f"{os.path.splitext(file_name)[0]}.png")
        if os.path.exists(mask_path):
            mask_np = np.array(Image.open(mask_path))
        else:
            mask_np = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)

        boundary_np = self._extract_boundary(mask_np)

        img_t, gt_t, mask_t, boundary_t = self.transform(
            img, gt_img, Image.fromarray(mask_np), Image.fromarray((boundary_np * 255).astype(np.uint8))
        )

        return {
            'image': img_t,
            'gt_image': gt_t,
            'gt_masks': mask_t,
            'gt_boundary': boundary_t,
            'gt_semantic_mask': mask_t,
            'file_name': file_name,
            'image_id': info['id']
        }

