import numpy as np
import quadprog
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward

from typing import TYPE_CHECKING, Optional, List
import torch
from torch.optim.lr_scheduler import _LRScheduler
import warnings

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from src.utils import get_grad_normL2

class LinearWarmup(_LRScheduler):
    def __init__(self, optimizer, total_iters, start_factor=0.0, end_factor=1.0, 
                 verbose=False):
        # sanity check parameters
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.last_iteration = 0
        self.initial_lr = [group["lr"] for group in optimizer.param_groups]
        super(LinearWarmup, self).__init__(optimizer, last_epoch=-1, verbose=verbose)

    def get_lr(self):
        if self.last_iteration > self.total_iters:
            return self.initial_lr
        lr_list = [(val * (self.last_iteration/self.total_iters)) #(self.end_factor-self.start_factor)
            for val in self.initial_lr]
        print("lr", lr_list)
        return lr_list
        
    def step(self, epoch=None):
        self.last_iteration += 1
        values = self.get_lr()
        
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

    def reset(self):
        self.last_iteration = 0
        

class LRWarmupScheduler(StrategyPlugin):
    def __init__(self, scheduler):
        super().__init__()

        self.scheduler = scheduler
        return

    def after_training_iteration(self, strategy, **kwargs):
        self.scheduler.step()
        return


    def after_training_exp(self, strategy, **kwargs):
        self.scheduler.reset()
        return