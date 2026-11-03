"""
TAS-UISNet: Robust Data Augmentations and Transforms for Degraded Underwater Vision
Author: Veeresh Birdar (SRM Institute of Science and Technology)
Target: Image and Vision Computing (Elsevier)
"""

import random
import torch
import numpy as np
import torchvision.transforms.functional as TF


class UnderwaterTransform:
    """
    Simultaneous transform for degraded underwater image, ground-truth reference,
    instance masks, and boundary maps.
    """
    def __init__(self, img_size=(480, 640), is_train=True):
        self.img_size = img_size
        self.is_train = is_train

    def __call__(self, img, gt_img=None, mask=None, boundary=None):
        # Resize to fixed dimension
        img = TF.resize(img, self.img_size)
        if gt_img is not None:
            gt_img = TF.resize(gt_img, self.img_size)
        if mask is not None:
            mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        if boundary is not None:
            boundary = TF.resize(boundary, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal and vertical flips during training
        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                if gt_img is not None:
                    gt_img = TF.hflip(gt_img)
                if mask is not None:
                    mask = TF.hflip(mask)
                if boundary is not None:
                    boundary = TF.hflip(boundary)

            if random.random() > 0.5:
                img = TF.vflip(img)
                if gt_img is not None:
                    gt_img = TF.vflip(gt_img)
                if mask is not None:
                    mask = TF.vflip(mask)
                if boundary is not None:
                    boundary = TF.vflip(boundary)

        # To Tensor and Normalization
        img_t = TF.to_tensor(img)
        gt_t = TF.to_tensor(gt_img) if gt_img is not None else img_t.clone()
        mask_t = TF.to_tensor(mask) if mask is not None else torch.zeros((1, self.img_size[0], self.img_size[1]))
        boundary_t = TF.to_tensor(boundary) if boundary is not None else torch.zeros((1, self.img_size[0], self.img_size[1]))

        return img_t, gt_t, mask_t, boundary_t

