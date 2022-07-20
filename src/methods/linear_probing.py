import numpy as np
import random
import copy
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from typing import TYPE_CHECKING, Optional, List

from avalanche.models import avalanche_forward
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.utils import freeze_everything, unfreeze_everything

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class LinearProbePlugin(StrategyPlugin):
    """
    Adding a linear probing stage between experiences
    """

    def __init__(self, num_epochs: int = 1):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """
        super().__init__()

        self.num_epochs = num_epochs
        return

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        """
        Doing the linear probing before the training of the experience.
            SideNote: The 'clock' counter will not be increased because the 'after_training_iteration' call
            is not made.
        """
        if strategy.clock.train_exp_counter > 0:
            # freeze the backbone
            freeze_everything(strategy.model.feature_extractor)
            # initially zero_grad the optimizer (to prevent nasty side effects)
            strategy.optimizer.zero_grad()
            for _ in range(self.num_epochs):
                # get the dataloader of the current experience
                for mbatch in strategy.dataloader:
                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                    x = x.to(strategy.device)
                    y = y.to(strategy.device)
                    out = avalanche_forward(strategy.model, x,
                                        tid)
                    loss = strategy._criterion(out, y) # divide loss by num updates to normalize gradient
                    loss.backward()

                    strategy.optimizer.step()
                    strategy.optimizer.zero_grad()
            # unfreeze the backbone
            unfreeze_everything(strategy.model.feature_extractor)
        return