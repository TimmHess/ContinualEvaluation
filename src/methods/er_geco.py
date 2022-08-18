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

from src.methods.replay import ERPlugin, ACE_CE_Loss


class GECOERPlugin(ERPlugin):
    def __init__(self, n_total_memories, num_tasks, device,
            alpha=0.99, lagrange_update_step=1, ace_ce_loss=False):

        super().__init__(n_total_memories, num_tasks, device=device, lmbda=0.5, 
                        ace_ce_loss=ace_ce_loss)

        self.constraint_ma = 0.0 # constraint moving average
        self.alpha = 0.99 # EMA constant
        self.kappa = 0.1 # loss discount value
        self.lagrange_update_step = 1 # update lagrange every n steps

        self.lagrange_lambda = 1.0
        return

    # TODO: set initial kappa to average loss on the replay memory after every completed task

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        """
        Code by: https://github.com/denproc/Taming-VAEs/blob/master/IWAE_GECO.py
        """
        # constraint is the loss on ER 
        if strategy.clock.train_exp_counter > 0:
            with torch.no_grad():
                replay_loss = self.replay_loss.detach() - self.kappa
                if self.constraint_ma is None:
                    self.constraint_ma = replay_loss
                else:
                    #print("constraint_ma:", self.constraint_ma, "replay_loss:", replay_loss)
                    self.constraint_ma = self.alpha * self.constraint_ma + (1-self.alpha) * replay_loss

                lambda_update_factor = replay_loss + (self.constraint_ma-replay_loss)
                #self.lagrange_lambda *= torch.clamp(torch.exp(lambda_update_factor), 0.9, 1.1)  # magic numbers taken from above code 
                lambda_update_factor = torch.exp(lambda_update_factor)
                #print("\n update_factor:", lambda_update_factor)
                self.lagrange_lambda *= lambda_update_factor  # magic numbers taken from above code 
                #print("resulting lambda=", self.lagrange_lambda)
        return 

    def before_backward(self, strategy, **kwargs):
        """
        Weight the current loss as well
        """

        strategy.loss = strategy.loss + self.lagrange_lambda * self.replay_loss # f(x) + lambda * (g(x) - constraint)
        return