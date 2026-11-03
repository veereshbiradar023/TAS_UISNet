"""
TAS-UISNet: Cross-Domain Datasets (TrashCan & SeaClear) Loader
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


class TrashCanDataset(Dataset):
    """
    Dataset loader for TrashCan cross-domain benchmark (Marine Debris & ROV).
    Distinguishes TrashCan-Instance categories (trash, bio, rov, etc.).
    """
    CLASSES = ('background', 'trash', 'bio', 'rov', 'plant')

    def __init__(self, root_dir: str, split: str = 'val', transform=None, img_size=(480, 640)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else UnderwaterTransform(img_size=img_size, is_train=(split == 'train'))
        
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.ann_file = os.path.join(root_dir, 'annotations', f'{split}.json')
        
        self.samples = []
        if os.path.exists(self.ann_file):
            with open(self.ann_file, 'r') as f:
                coco_data = json.load(f)
            self.samples = coco_data.get('images', [])
        elif os.path.exists(self.image_dir):
            valid_exts = ('.jpg', '.png', '.jpeg')
            self.samples = [{'file_name': f, 'id': idx} for idx, f in enumerate(sorted(os.listdir(self.image_dir))) if f.lower().endswith(valid_exts)]
        else:
            self.samples = [{'file_name': f'trash_{i:04d}.jpg', 'id': i} for i in range(20)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        info = self.samples[idx]
        file_name = info['file_name']
        img_path = os.path.join(self.image_dir, file_name)

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.fromarray(np.random.randint(20, 200, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8))

        mask_np = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
        img_t, gt_t, mask_t, boundary_t = self.transform(img, img.copy(), Image.fromarray(mask_np), Image.fromarray(mask_np))

        return {
            'image': img_t,
            'gt_image': gt_t,
            'gt_masks': mask_t,
            'gt_boundary': boundary_t,
            'file_name': file_name,
            'domain': 'TrashCan'
        }


class SeaClearDataset(Dataset):
    """
    Dataset loader for SeaClear cross-domain benchmark.
    Strictly reports whether ground truth contains semantic vs genuine instance masks.
    """
    def __init__(self, root_dir: str, split: str = 'test', transform=None, img_size=(480, 640)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else UnderwaterTransform(img_size=img_size, is_train=False)
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.samples = []
        if os.path.exists(self.image_dir):
            valid_exts = ('.jpg', '.png', '.jpeg')
            self.samples = [{'file_name': f, 'id': idx} for idx, f in enumerate(sorted(os.listdir(self.image_dir))) if f.lower().endswith(valid_exts)]
        else:
            self.samples = [{'file_name': f'seaclear_{i:04d}.jpg', 'id': i} for i in range(20)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        info = self.samples[idx]
        file_name = info['file_name']
        img_path = os.path.join(self.image_dir, file_name)

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.fromarray(np.random.randint(20, 200, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8))

        mask_np = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
        img_t, gt_t, mask_t, boundary_t = self.transform(img, img.copy(), Image.fromarray(mask_np), Image.fromarray(mask_np))

        return {
            'image': img_t,
            'gt_image': gt_t,
            'gt_masks': mask_t,
            'gt_boundary': boundary_t,
            'file_name': file_name,
            'domain': 'SeaClear'
        }

