from collections import defaultdict

import warnings
import torch
from torch.utils.data import DataLoader
from copy import copy, deepcopy

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict

from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy

import numpy as np

class AGEM_BGD(BaseStrategy):
    """
    A BATCH GRADIETN DESCENT implementation using the AGEM continual learning method.
    """
    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):

        super().__init__(model, optimizer,
                 criterion,
                 train_mb_size, train_epochs,
                 eval_mb_size, device,
                 plugins,
                 evaluator, eval_every)

        self.prev_exp_dataloaders = []

        self.reference_gradients = None
        self.eps_agem = 1e-7

        self.use_agem = True
        self.use_replay = True
        if self.use_agem:
            print("Using BGD with AGEM")
        if self.use_replay:
            print("Using BGD with replay")

        return

    def _before_training_epoch(self, **kwargs):
        """
        Called at the beginning of a new training epoch.

        AGEM update will now be calculated here because iterations are replaced by epochs.

        :param kwargs:
        :return:
        """
        # Reset optimizer for next epoch's gradient accumulation
        self.optimizer.zero_grad()

        # Replay
        print("num accum dataloaders:", len(self.prev_exp_dataloaders))
        if self.use_replay:
            # backward loss for all dataloaders
            for d_it, dloader in enumerate(self.prev_exp_dataloaders):
                print("forwarding loss for dataset {}".format(d_it))
                #dataloader = DataLoader(dset, batch_size=self.train_mb_size)
                for i, mbatch in enumerate(dloader):
                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
                    if i == 0:
                        print(np.unique(y))

                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = avalanche_forward(self.model, x,
                                        tid)
                    loss = self._criterion(out, y) / len(dloader) # divide loss by num updates to normalize gradient
                    loss.backward()
            # AGEM
            if self.use_agem and self.clock.train_exp_counter > 0:
                self.reference_gradients = [
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=self.device)
                    for n, p in self.model.named_parameters()
                ]
                self.reference_gradients = torch.cat(self.reference_gradients)
        # print("after replay:") 
        # for n, p in self.model.named_parameters():
        #     print(p.grad)

        # Finally, call assigned plugins
        super()._before_training_epoch(**kwargs)


    def training_epoch(self, **kwargs):
        for i, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            
            self.loss = 0

            # NO ZERO_GRAD

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion() / len(self.dataloader) # divide loss by num updates to normalize gradient

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # NO ZERO_GRAD

            self._after_training_iteration(**kwargs)
        
        # print("after epoch:") 
        # for n, p in self.model.named_parameters():
        #     print(p.grad)
        return


    def _after_training_epoch(self, **kwargs):
        # Optimization step
        self._before_update(**kwargs)
        self.optimizer.step()
        self._after_update(**kwargs)

        super()._after_training_epoch(**kwargs)
        return 


    def _after_training_exp(self, **kwargs):
        """
        Save a copy of the model after each experience
        """
        dl_copy = copy(self.dataloader)
        self.prev_exp_dataloaders.append(dl_copy)

        super()._after_training_exp(**kwargs)
        return

    @torch.no_grad()
    def _after_backward(self, **kwargs):
        """
        Project gradient based on reference gradients 
        @ copied avalanche code
        """
        if self.use_agem:
            if not self.reference_gradients is None:
                current_gradients = [
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=self.device)
                    for n, p in self.model.named_parameters()
                ]
                current_gradients = torch.cat(current_gradients)

                assert (
                    current_gradients.shape == self.reference_gradients.shape
                ), "Different model parameters in AGEM projection"

                dotg = torch.dot(current_gradients, self.reference_gradients)
                if dotg < 0:
                    alpha2 = dotg / (
                        torch.dot(self.reference_gradients, self.reference_gradients) 
                        + self.eps_agem
                    )
                    
                    grad_proj = (
                        current_gradients - self.reference_gradients * alpha2
                    )

                    count = 0
                    for n, p in self.model.named_parameters():
                        n_param = p.numel()
                        if p.grad is not None:
                            p.grad.copy_(
                                grad_proj[count : count + n_param].view_as(p)
                            )
                        count += n_param
        super()._after_backward(**kwargs)
        return
       
    