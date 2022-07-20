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
from tqdm import tqdm

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

from avalanche.training.storage_policy import BalancedExemplarsBuffer, ReservoirSamplingBuffer
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset, AvalancheConcatDataset


if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class ACE_CE_Loss(nn.Module):
    """
    Masked version of CrossEntropyLoss.
    """
    def __init__(self, device):
        super(ACE_CE_Loss, self).__init__()

        self.seen_so_far = torch.zeros(0, dtype=torch.int).to(device) # basically an empty tensor
        return

    def forward(self, logits, labels):
        present = labels.unique()

        mask = torch.zeros_like(logits).fill_(-1e9)
        mask[:, present] = logits[:, present] # add the logits for the currently observed labels
        
        if len(self.seen_so_far) > 0: # if there are seen classes, add them as well (this is for replay buffer loss)
            mask[:, (self.seen_so_far.max()+1):] = logits[:, (self.seen_so_far.max()+1):] # add the logits for the unseen labels
        
        logits = mask
        return F.cross_entropy(logits, labels)

    def update_seen(self, labels):
        self.seen_so_far = torch.cat([self.seen_so_far, labels]).unique()
        return


class DERClassBalancedBuffer(BalancedExemplarsBuffer):
    """ Stores samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each
        class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True,
                 total_num_classes: int = None):
        """
        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()
        
        self.logits_buffer = {}


    def update(self, strategy: "BaseStrategy", **kwargs):
        print("DERReplay: Updating buffer")
        new_data = strategy.experience.dataset
        # TODO: make sure there is no transform active in the new_dataset!
        
        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheSubset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy,
                                                class_to_len[class_id])
        
        # Add logits to targets for each cl_dataset
        strategy.model.eval()
        for class_id, class_buf in self.buffer_groups.items():
            #print("id:", class_id, "ClassBuf:", class_buf.buffer) # class_buf.buffer is an AvalancheDataset
            # if there is no entry for the class_id -> add its logits to the buffer
            if not class_id in self.logits_buffer:
                print("Preparing logits for class {}".format(class_id))
                self.logits_buffer[class_id] = []
                # Run through 'new_data' and compute the logits for each example
                new_data_loader = DataLoader(class_buf.buffer, batch_size=1, shuffle=False) 
                for i, data in tqdm(enumerate(new_data_loader)):
                    img, _, _ = data
                    img = img.to(strategy.device)
                    logits = strategy.model.feature_extractor(img)
                    logits = logits.detach().squeeze(0).cpu()
                    self.logits_buffer[class_id].append(logits)
            else:
                print("Logits for class {} already exist".format(class_id))
        strategy.model.train()
        return        

class DERPlugin(StrategyPlugin):
    """
    Rehearsal Revealed: replay plugin.
    Implements two modes: Classic Experience Replay (ER) and Experience Replay with Ridge Aversion (ERaverse).
    """
    store_criteria = ['rnd']

    def __init__(self, n_total_memories, num_tasks, device,
        lmbda: float, lmbda_warmup_steps=0, do_decay_lmbda=False, ace_ce_loss=False):
        """
        Standard samples the same batch-size of new samples.

        :param n_total_memories: The maximal number of input samples to store in total.
        :param num_tasks:        The number of tasks being seen in the scenario.
        :param mode:             'ER'=regular replay, 'ERaverse'=Replay with Ridge Aversion.
        :param init_epochs:      Number of epochs for the first experience/task.
        """
        super().__init__()

        # Memory
        self.n_total_memories = n_total_memories  # Used dynamically
        self.num_tasks = num_tasks
        # a Dict<task_id, Dataset>
        # self.storage_policy = ClassBalancedBuffer(  # Samples to store in memory
        #     max_size=self.n_total_memories,
        #     adaptive_size=True,
        # )
        self.storage_policy = DERClassBalancedBuffer(
            max_size=self.n_total_memories,
            adaptive_size=True,
        )
        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        print(f"[METHOD CONFIG] SUMMARY: ", end='')
        pprint(self.__dict__, indent=2)

        # device
        self.device = device

        # weighting of replayed loss and current minibatch loss
        self.lmbda = lmbda  # 0.5 means equal weighting of the two losses
        self.do_decay_lmbda = do_decay_lmbda
        self.lmbda_warmup_steps = lmbda_warmup_steps
        self.do_exp_based_lmbda_weighting = False
        self.last_iteration = 0

        # Losses
        self.replay_criterion = torch.nn.CrossEntropyLoss()
        self.der_criterion = torch.nn.MSELoss()
        self.use_ace_ce_loss = ace_ce_loss
        if self.use_ace_ce_loss:
            self.replay_criterion = ACE_CE_Loss(self.device)
            self.ace_ce_loss = ACE_CE_Loss(self.device)
        self.replay_loss = 0


    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        #import numpy as np
        #print("num classes:", len(np.unique(strategy.experience.dataset.targets)))
        if strategy.clock.train_exp_counter > 0 and self.do_decay_lmbda:
            lmbda_decay_factor = (strategy.clock.train_exp_counter) / (strategy.clock.train_exp_counter+1)
            print("\nDecaying lmbda by:", lmbda_decay_factor)
            self.lmbda *= lmbda_decay_factor
            print("New lmbda is:", self.lmbda)
        return


    def before_training_iteration(self, strategy, **kwargs):
        """
        Adjust the lmbda weighting according to lmbda warmup settings
        """
        # Update ACE loss 'seen_so_far' attribute
        # if self.use_ace_ce_loss:
        #     self.replay_criterion.update_seen(strategy.mb_y)
        #     print("\nreplay seen_so_far of replay loss:", self.replay_criterion.seen_so_far)
        #     print("ace seen_so_far:", self.ace_ce_loss.seen_so_far)
            

        #lmbda_weighting stays 1.0 for the first experience
        if strategy.clock.train_exp_counter > 0:
            self.last_iteration += 1
            #print("before_iteration:", self.last_iteration, self.lmbda_warmup_steps)
            if not self.last_iteration > self.lmbda_warmup_steps:
                # Apply linear weighting over the number of warmup steps
                self.lmbda_weighting = self.last_iteration / self.lmbda_warmup_steps
                #print("lmbda_weighting is:", self.lmbda_weighting)
        return


    def before_forward(self, strategy, **kwargs):
        """
        Calculate the loss with respect to the replayed data separately here.
        This enables to weight the losses separately.
        Needs to be done here to prevent gradients being zeroed!
        """
        # Sample memory batch
        x_s, y_s, t_s = None, None, None
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:  # Only sample if there are stored
            x_s, y_s, t_s, l_s = self.load_buffer_batch(storage_policy=self.storage_policy, 
                                        strategy=strategy, nb=strategy.train_mb_size)
        
        # Run forward for replayed data separately
        strategy.model.train()
        # reset replay_loss (even though this might not be necessary)
        self.replay_loss = 0
        if x_s is not None:  
            assert y_s is not None
            assert t_s is not None
            assert l_s is not None
            l_s = l_s.to(strategy.device)

            out = avalanche_forward(strategy.model, x_s, t_s)
            out_logits = strategy.model.last_features
            print(out_logits.shape, l_s.shape)
            
            # TODO: Also calculate the DER loss, respectivley ONLY calculate the DER loss
            self.replay_loss = self.der_criterion(out_logits, l_s)
            import sys; sys.exit()  
            #loss = self.replay_criterion(out, y_s)
            #self.replay_loss = self.replay_criterion(out, y_s)
            #loss *= ((1-self.lmbda)*2)  # apply weighting // the *2 is to make up for the lambda factor
            #self.replay_loss = loss
            #loss.backward() # backward is done in global backward -> loss is taken into account in 'before_backward'
        return


    def before_backward(self, strategy, **kwargs):
        """
        Weight the current loss as well
        """
        # overwrite loss (this is certainly not the best solution - this should be done in the strategy not the plugin)
        # this way, currently, the loss is calculated two times...
        if self.use_ace_ce_loss:
            #strategy.loss = 0
            strategy.loss = self.ace_ce_loss(strategy.mb_output, strategy.mb_y)
            
        if strategy.clock.train_exp_counter > 0:
            #print("lmbda weighting:", self.lmbda_weighting)
            strategy.loss *= self.lmbda_weighting # used in lmbda_warmup

        strategy.loss = self.lmbda * strategy.loss + (1-self.lmbda) * self.replay_loss
        return

    def after_training_exp(self, strategy, **kwargs):
        """ Update memories."""
        self.storage_policy.update(strategy, **kwargs)  # Storage policy: Store the new exemplars in this experience
        self.reset()
        return

    def reset(self):
        """
        Reset internal variables after each experience
        """
        self.last_iteration = 0
        self.lmbda_weighting = 1
        return

    def load_buffer_batch(self, storage_policy, strategy, nb=None):
        """
        Wrapper to retrieve a batch of exemplars from the rehearsal memory
        :param nb: Number of memories to return
        :return: input-space tensor, label tensor
        """
        ret_x, ret_y, ret_t, ret_l = None, None, None, None
        # Equal amount as batch: Last batch can contain fewer!
        n_exemplars = strategy.train_mb_size if nb is None else nb
        new_dset, logits_set = self.retrieve_random_buffer_batch(storage_policy, n_exemplars)  # Dataset object

        # Load the actual data
        logits_batch_start = 0
        logits_batch_end = 0
        for sample in DataLoader(new_dset, batch_size=len(new_dset), pin_memory=True, shuffle=False):
            x_s, y_s = sample[0].to(strategy.device), sample[1].to(strategy.device)
            t_s = sample[-1].to(strategy.device)  # Task label (for multi-head)
            
            # logits_batch_end = logits_batch_start + len(y_s)
            # l_s = logits_set[logits_batch_start:logits_batch_end]
            # logits_batch_start = logits_batch_end
            # in the current implementation, this is valid as well - if multiple batches are loaded use above
            l_s = logits_set

            ret_x = x_s if ret_x is None else torch.cat([ret_x, x_s])
            ret_y = y_s if ret_y is None else torch.cat([ret_y, y_s])
            ret_t = t_s if ret_t is None else torch.cat([ret_t, t_s])
            ret_l = l_s if ret_l is None else torch.cat([ret_l, l_s])
        return ret_x, ret_y, ret_t, ret_l

    def retrieve_random_buffer_batch(self, storage_policy, n_samples):
        """
        Retrieve a batch of exemplars from the rehearsal memory.
        First sample indices for the available tasks at random, then actually extract from rehearsal memory.
        There is no resampling of exemplars.

        :param n_samples: Number of memories to return
        :return: input-space tensor, label tensor
        """
        assert n_samples > 0, "Need positive nb of samples to retrieve!"

        # Determine how many mem-samples available
        q_total_cnt = 0  # Total samples
        free_q = {}  # idxs of which ones are free in mem queue
        tasks = []
        for t, ex_buffer in storage_policy.buffer_groups.items():
            mem_cnt = len(ex_buffer.buffer)  # Mem cnt
            free_q[t] = list(range(0, mem_cnt))  # Free samples
            q_total_cnt += len(free_q[t])  # Total free samples
            tasks.append(t)

        # Randomly sample how many samples to idx per class
        free_tasks = copy.deepcopy(tasks)
        tot_sample_cnt = 0
        sample_cnt = {c: 0 for c in tasks}  # How many sampled already
        max_samples = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
        while tot_sample_cnt < max_samples:
            t_idx = random.randrange(len(free_tasks))
            t = free_tasks[t_idx]  # Sample a task

            if sample_cnt[t] >= len(storage_policy.buffer_group(t)):  # No more memories to sample
                free_tasks.remove(t)
                continue
            sample_cnt[t] += 1
            tot_sample_cnt += 1

        # Actually sample
        s_cnt = 0
        subsets = []
        logit_subsets = []
        for t, t_cnt in sample_cnt.items():
            if t_cnt > 0:
                # Set of idxs
                cnt_idxs = torch.randperm(len(storage_policy.buffer_group(t)))[:t_cnt]
                sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
                sample_idxs = sample_idxs.view(-1)
                
                # Select the logits accoding to the same 'sample_idxs' set
                logits_subset = torch.stack(storage_policy.logits_buffer[t])[sample_idxs]
                print("logits_subset", logits_subset.shape)
                logit_subsets.append(logits_subset)

                # Actual subset
                s = Subset(storage_policy.buffer_group(t), sample_idxs.tolist())
                subsets.append(s)
                s_cnt += t_cnt
        assert s_cnt == tot_sample_cnt == max_samples
        new_dset = ConcatDataset(subsets)
        logit_subsets = torch.cat(logit_subsets, dim=0)

        return new_dset, logit_subsets
