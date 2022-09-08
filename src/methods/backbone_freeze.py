import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from typing import TYPE_CHECKING, Optional, List
from typing import NamedTuple, List, Optional, Tuple, Callable

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import freeze_everything
from src.utils import get_grad_normL2

from src.model import freeze_up_to


class FreezeBackbonePlugin(StrategyPlugin):
    def __init__(self, exp_to_freeze_on=0, freeze_up_to_layer_name=None):
        super().__init__()

        self.exp_to_freeze_on = exp_to_freeze_on
        self.freeze_up_to_layer_name = freeze_up_to_layer_name
        return

    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > (self.exp_to_freeze_on -1): # NOTE: -1 is required to be able to freeze on the 0th experience
            print("\n\nFreezing backbone...\n\n")
            if self.freeze_up_to_layer_name is None:
                print("Freezing entire model...")
                freeze_everything(strategy.model.feature_extractor)
            else: 
                print("Freezing model up to layer {}...".format(self.freeze_up_to_layer_name))
                frozen_layers, _ = freeze_up_to(strategy.model.feature_extractor, self.freeze_up_to_layer_name)
                for layer_name in frozen_layers:
                    print("Froze layer: {}".format(layer_name))
        return