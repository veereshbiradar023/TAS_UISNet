"""
TAS-UISNet datasets package
"""
from .transforms import UnderwaterTransform
from .urpc_dataset import URPC2020Dataset
from .trashcan_dataset import TrashCanDataset, SeaClearDataset

__all__ = [
    'UnderwaterTransform',
    'URPC2020Dataset',
    'TrashCanDataset',
    'SeaClearDataset'
]

