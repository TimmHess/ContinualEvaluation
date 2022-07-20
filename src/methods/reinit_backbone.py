import numpy as np
import quadprog
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward

from typing import TYPE_CHECKING, Optional, List
import torch
import torch.nn as nn

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from src.utils import get_grad_normL2


class ReInitBackbonePlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()

        return

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
    

    def before_training_exp(self, strategy, **kwargs):
        """
        After the first experience, re-initialize weights of the model
        """
        if strategy.clock.train_exp_counter > 0:
            strategy.model.apply(self.initialize_weights)
            print("\nRe-Initialized weights in the backbone!\n")
        return