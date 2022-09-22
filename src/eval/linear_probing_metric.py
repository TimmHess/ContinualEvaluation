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
from avalanche.models.dynamic_modules import MultiHeadClassifier, IncrementalClassifier

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
    def __init__(self, train_stream, eval_all=False, force_task_eval=False,
            num_finetune_epochs=1, batch_size=32, num_workers=0):
        self._accuracy = Accuracy() # metric calculation container
        super(LinearProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='experience', emit_at='experience',
            mode='eval')

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_stream = train_stream
        self.num_exps_seen = 0
        self.num_fintune_epochs = num_finetune_epochs

        self.head_copy = None # NOTE: local copy of the model's head used for linear probing
        self.local_optim = None

        self.training_complete = False

        self.eval_all = eval_all # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.force_task_eval = force_task_eval # flag to indicate forced evaluation on all experiences for the current task (including yet unseed ones)
        self.initial_out_features = None
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
            if self.force_task_eval:
                result = self._metric.result(torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc')) 
                return result
            
            return self._metric.result(phase_and_task(strategy)[1]) # NOTE: phase_and_task potential source of errors in logging!

    def update(self, strategy=None):
        # Get task labels defined for each experience           # NOTE: Why was this necessary in the first place?
        # task_labels = strategy.experience.task_labels
        # if len(task_labels) > 1:
        #     # task labels defined for each pattern
        #     task_labels = strategy.mb_task_id
        # else:
        #     task_labels = task_labels[0]

        task_labels = strategy.mb_task_id
        y = strategy.mb_y
        # print("\nTask_labels:", task_labels)
        # print("Y:")
        # print(y)

        # Adjust task_labels if 'forced_task_eval' flag active
        if self.force_task_eval:
            task_labels = torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc') # equivalent to '//' operation
            y = y % self.initial_out_features

            # print("\nTask_labels:", task_labels)
            # print("Y:")
            # print(y)

        # Get representation of current mbatch from backbone
        x_rep = strategy.model.last_features.detach()
    
        # Forward representation through new head (linear probe)
        if isinstance(self.head_copy, MultiTaskModule): # NOTE: this is copied from the 'avalanche_forward'-function 
            out = self.head_copy(x_rep, task_labels) # shouldn_t this be (x_rep, strategy.mb_task_id) ?
        else:  # no task labels
            out = self.head_copy(x_rep)
        
        # Update the accuracy measure

        self._accuracy.update(out, y, task_labels) # TODO: replace task_labels with strategy.mb_task_id?s
        #print("\n", self._accuracy.result())
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
                # Check number of current heads against max numbre of heads possible
                self.head_copy = copy.deepcopy(strategy.model.classifier)
                print("\nis multi-headed:", isinstance(self.head_copy, MultiTaskModule))

                #######
                if self.force_task_eval: # NOTE: Re-initialize self.head_copy as MultiTaskModule for task-wise evaluation
                    print("Forcing multi-headed eval!")
                    feat_size = self.head_copy.in_features
                    self.initial_out_features = len(torch.unique(torch.tensor(self.train_stream[0].dataset.targets))) # NOTE: not the best way because it assumes that the first experience is representative
                    lin_bias = self.head_copy.bias is not None
                    print("\nReplaced head_copy with MultiHeadModule using feat_size: {} and initial_out_features: {}".format(feat_size, self.initial_out_features))
                    self.head_copy = MultiHeadClassifier(in_features=feat_size,
                                             initial_out_features=self.initial_out_features,
                                             use_bias=lin_bias)
                    # Force adapation of MultiHeadClassifier by hand.. NOTE: this is needed because original adaptation method does not work with CI datasets
                    for exp_id, _ in enumerate(self.train_stream):
                        tid = str(exp_id)  # need str keys
                        if tid not in self.head_copy.classifiers:
                            new_head = IncrementalClassifier(feat_size, self.initial_out_features)
                            self.head_copy.classifiers[tid] = new_head
                            #print("\nAdding new heads to Linear Probe")
                #######
                
                if isinstance(self.head_copy, MultiTaskModule): # NOTE: this adds classifiers for every task possible
                    if len(self.head_copy.classifiers) < len(self.train_stream):
                        for exp in self.train_stream:
                            self.head_copy.adaptation(exp.dataset)
   
                # Move novel probe head to common device and (re-)initialize
                self.head_copy = self.head_copy.to(strategy.device)
                print("Reinitializing weights...")
                initialize_weights(self.head_copy)
                self.head_copy.train() # set to train mode (for safety)
                
                # Initialize local optimizer for the new head
                #self.local_optim = torch.optim.Adam(self.head_copy.parameters(), lr=0.01, weight_decay=0.0, betas=(0.9, 0.999))
                self.local_optim = torch.optim.AdamW(self.head_copy.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999))
                # Prepare dataet and dataloader
                if self.eval_all: # NOTE: Override the number of experiences to use in each step with max value
                    self.num_exps_seen = len(self.train_stream) -1 # -1 to make up for +1 in next step
                    print("\nNum seen experiences is maxed out!")

                curr_exp_data_stream = self.train_stream[:(self.num_exps_seen+1)] # Grab the curent subset of experiences from train_stream
                curr_exp_data = []
                # Create a ConcatDataset and respective Dataloader
                for exp in curr_exp_data_stream:
                    curr_exp_data.append(exp.dataset)
                    # Update head_copy for multi-task setting                           # NOTE: Redundant code.. why was this used in the first place? 
                    # if self.eval_all and isinstance(self.head_copy, MultiTaskModule):
                    #     self.head_copy.adaptation(exp.dataset)
                lp_dataset = torch.utils.data.ConcatDataset(curr_exp_data)
                lp_dataloader = torch.utils.data.DataLoader(lp_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers, drop_last=True) 
                print("Collected dataset and loader...")
                
                # Train the new head(s)
                print("Training new head(s)...", "num heads:", len(self.head_copy.classifiers))
                for _ in tqdm(range(self.num_fintune_epochs)):
                    for _, mbatch in enumerate(lp_dataloader):
                        self.local_optim.zero_grad()

                        x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                        # On-the-fly update labels and targets for task-incremental learning
                        if self.force_task_eval: 
                            y, tid = y % self.initial_out_features, torch.div(y, self.initial_out_features, rounding_mode='trunc') # NOTE: This assumes that the number of classes per head is constant!

                        # print("targets:", y)
                        # print("task id:", tid)

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


    