#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import random
import copy
from pprint import pprint
from typing import TYPE_CHECKING, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from avalanche.models import avalanche_forward
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.storage_policy import ClassBalancedBuffer

from src.utils import get_grad_normL2
from src.eval.continual_eval import ContinualEvaluationPhasePlugin

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class EpochLengthAdapterPlugin(StrategyPlugin):
    
    def __init__(self, epochs):
        """
        """
        super().__init__()

        self.epochs = epochs
        return

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if strategy.clock.train_exp_counter < len(self.epochs):
            strategy.train_epochs = self.epochs[strategy.clock.train_exp_counter]
            print("\nUsing", self.epochs[strategy.clock.train_exp_counter], " epochs during experience", strategy.clock.train_exp_counter, "\n")
        return
