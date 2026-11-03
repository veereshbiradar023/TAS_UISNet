"""
TAS-UISNet losses package
"""
from .total_loss import (
    CharbonnierLoss,
    SSIMLoss,
    ColorAndFrequencyLoss,
    HeteroscedasticUncertaintyLoss,
    BoundaryLoss,
    TAS_UISNet_TotalLoss
)

__all__ = [
    'CharbonnierLoss',
    'SSIMLoss',
    'ColorAndFrequencyLoss',
    'HeteroscedasticUncertaintyLoss',
    'BoundaryLoss',
    'TAS_UISNet_TotalLoss'
]

