import numpy as np
import quadprog
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward

from typing import TYPE_CHECKING, Optional, List
import torch

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from src.utils import get_grad_normL2


class GradClipPlugin(StrategyPlugin):
    def __init__(self, clip_value: float):
        super().__init__()

        self.clip_value = clip_value
        return

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Apply gradient clipping to all gradients of the model.
        """

        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), max_norm=self.clip_value,
            norm_type=2, error_if_nonfinite=True)
        return