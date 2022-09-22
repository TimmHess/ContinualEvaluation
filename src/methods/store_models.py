import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from typing import TYPE_CHECKING, Optional, List
from typing import NamedTuple, List, Optional, Tuple, Callable

from pathlib import Path

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import freeze_everything, get_layers_and_params
from src.utils import get_grad_normL2


class StoreModelsPlugin(StrategyPlugin):
    def __init__(self, model_name, model_store_path):
        super().__init__()

        self.model_name = model_name
        self.model_store_path = model_store_path
        return

    def after_training_exp(self, strategy, **kwargs):
        # Store model backbone to path
        dir_path = str(self.model_store_path) + "/model_weights/"
        file_name =  self.model_name + "_" + str(strategy.clock.train_exp_counter) + ".pth"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        torch.save(strategy.model.state_dict(), dir_path+file_name)
        print("\nStoring model to path: ", (dir_path+file_name))
        return 