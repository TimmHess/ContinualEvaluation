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
            alpha=0.99, lagrange_update_set=1, ace_ce_loss=False):

        super().__init__(n_total_memories, num_tasks, device=device, lmbda=0.5, 
                        ace_ce_loss=ace_ce_loss)

        self.constraint_ma = None
        self.alpha = 0.99
        self.lagrange_update_set = 1
        self.kappa = 0.1

        self.lagrange_lambda = 1.0 # is now handled as Lagrange multiplier
        self.lmbda_warmup_steps = 0
        self.do_decay_lmbda = False,
        return

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        """
        Code by: https://github.com/denproc/Taming-VAEs/blob/master/IWAE_GECO.py
        """
        # constraint is the loss on ER 
        if strategy.clock.train_exp_counter > 0:
            with torch.no_grad():
                replay_loss = self.replay_loss.detach() - (self.kappa**2)
                if self.constraint_ma is None:
                    self.constraint_ma = replay_loss
                else:
                    print("constraint_ma:", self.constraint_ma, "replay_loss:", replay_loss)
                    self.constraint_ma = self.alpha * self.constraint_ma + (1-self.alpha) * replay_loss

                lambda_update_factor = replay_loss + (self.constraint_ma-replay_loss)
                #self.lagrange_lambda *= torch.clamp(torch.exp(lambda_update_factor), 0.9, 1.1)  # magic numbers taken from above code 
                lambda_update_factor = torch.exp(lambda_update_factor)
                print("\n update_factor:", lambda_update_factor)
                self.lagrange_lambda *= lambda_update_factor  # magic numbers taken from above code 
                print("resulting lambda=", self.lagrange_lambda)
        return 

    def before_backward(self, strategy, **kwargs):
        """
        Weight the current loss as well
        """
        # overwrite loss (this is certainly not the best solution - this should be done in the strategy not the plugin)
        # this way, currently, the loss is calculated two times...
        # if self.use_ace_ce_loss:
        #     #strategy.loss = 0
        #     strategy.loss = self.ace_ce_loss(strategy.mb_output, strategy.mb_y)
            
        # if strategy.clock.train_exp_counter > 0:
        #     #print("lmbda weighting:", self.lmbda_weighting)
        #     strategy.loss *= self.lmbda_weighting # used in lmbda_warmup

        strategy.loss = strategy.loss + self.lagrange_lambda * self.replay_loss # f(x) + lambda * g(x) - constraint
        return