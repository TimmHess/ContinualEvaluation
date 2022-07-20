import numpy as np
import quadprog
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward

from typing import TYPE_CHECKING, Optional, List
import torch

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import freeze_everything
from src.utils import get_grad_normL2


class FreezeBackbonePlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()
        return

    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            print("\n\nFreezing backbone...\n\n")
            freeze_everything(strategy.model.feature_extractor)
        return