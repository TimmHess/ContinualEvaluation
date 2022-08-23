#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
from collections import deque, defaultdict
import torch

from avalanche.models.dynamic_modules import MultiTaskModule

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.loss import Loss
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task
from avalanche.models.utils import avalanche_forward

from src.utils import get_grad_normL2
from src.model import initialize_weights

import copy

from tqdm import tqdm

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

class LinearProbingAccuracyMetric(GenericPluginMetric[float]):
    def __init__(self, train_stream, eval_all=False,
            num_finetune_epochs=1, batch_size=32, num_workers=0):
        self._accuracy = Accuracy() # metric calculation container
        super(LinearProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='experience', emit_at='experience',
            mode='eval')

        self.eval_all = eval_all # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_stream = train_stream
        self.num_exps_seen = 0
        self.num_fintune_epochs = num_finetune_epochs

        self.head_copy = None # NOTE: local copy of the model's head used for linear probing
        self.local_optim = None

        self.training_complete = False
        return

    def __str__(self):
        return "Top1_LP_Acc_Exp"

    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy=None):
        # Get task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
    
        # Get representation of current mbatch from backbone
        x_rep = strategy.model.last_features.detach()
    
        # Forward representation through new head (linear probe)
        if isinstance(self.head_copy, MultiTaskModule): # NOTE: this is the avalanche_forward function copied
            out = self.head_copy(x_rep, task_labels)
        else:  # no task labels
            out = self.head_copy(x_rep)
        
        # Update the accuracy measure    
        print(task_labels)
        self._accuracy.update(out, strategy.mb_y, task_labels)
        return


    def before_eval_exp(self, strategy: 'BaseStrategy'):
        # Check if LinearProbe already trained
        if not self.training_complete:
            # Set flag that will prevent retraining of LinearProbe for each sub-task
            self.training_complete = True
            print("\nLocked LinearProbe Training")
            # Initialize and prepare the linear probing head
            with torch.enable_grad(): # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
                print("\nPreparing Linear Probe(s)")
                print("Initializing new head(s)...")
                self.head_copy = copy.deepcopy(strategy.model.classifier)
                # Check number of current heads against max numbre of heads possible
                if isinstance(self.head_copy, MultiTaskModule):
                    print(len(self.train_stream))
                    if len(self.head_copy.classifiers) < len(self.train_stream):
                        print("\nAdding new heads to Linear Probe")
                        for exp in self.train_stream:
                            self.head_copy.adaptation(exp.dataset)
                self.head_copy = self.head_copy.to(strategy.device)
                print("Reinitializing weights...")
                initialize_weights(self.head_copy)
                self.head_copy.train()
                
                # Initialize local optimizer for the new head
                self.local_optim = torch.optim.Adam(self.head_copy.parameters(), lr=0.01, weight_decay=0.0, betas=(0.9, 0.999))
        
                # Prepare dataet and dataloader
                if self.eval_all: # NOTE: Override the number of experiences to use in each step with max value
                    self.num_exps_seen = len(self.train_stream) -1 # -1 to make up for +1 in next step
                    print("\nNum seen experiences is maxed out!")
                curr_exp_data_stream = self.train_stream[:(self.num_exps_seen+1)]
                curr_exp_data = []
                for exp in curr_exp_data_stream:
                    curr_exp_data.append(exp.dataset)
                    # Update head_copy for multi-task setting
                    if self.eval_all and isinstance(self.head_copy, MultiTaskModule):
                        self.head_copy.adaptation(exp.dataset)
                                
                lp_dataset = torch.utils.data.ConcatDataset(curr_exp_data)
                lp_dataloader = torch.utils.data.DataLoader(lp_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers) # TODO: copy the dataloaders hyper-params from cmd_parser...
                print("Collected dataset and loader...")
                
                # Train the new heads for n_ft epochs on the respective task train sets (allowing access to all previous data)
                #print(len(self.prev_exp_dataloaders)) # sanity check
                #for d_it, dloader in enumerate(self.prev_exp_dataloaders):
                #    print("\nTraining head on data from task {}\n".format(d_it))
                print("Training new head(s)...")
                for _ in tqdm(range(self.num_fintune_epochs)):
                    for _, mbatch in enumerate(lp_dataloader):
                        self.local_optim.zero_grad()

                        x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                        x = x.to(strategy.device)
                        y = y.to(strategy.device)
                        
                        # Get representation from backbone
                        x_rep = strategy.model.feature_extractor(x).detach() # detach() is most likely not necessary here but I want to be sure

                        # Forward representation through new head 
                        if isinstance(self.head_copy, MultiTaskModule): # NOTE: this is the avalanche_forward function copied
                            out = self.head_copy(x_rep, tid)
                        else:  # no task labels
                            out = self.head_copy(x_rep)

                        loss = strategy._criterion(out, y)
                        loss.backward()
                        self.local_optim.step()
                print("\nLinear Probe training complete...")

        super().before_eval_exp(strategy)
        if self._reset_at == 'experience' and self._mode == 'eval':
            self.reset(strategy)


    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Release the lock on LinearProbe training
        self.training_complete = False
        print("\nReleased Flag for Linear Probe Training")
        # Increase the counter on seen experiences
        self.num_exps_seen += 1 
        return


    