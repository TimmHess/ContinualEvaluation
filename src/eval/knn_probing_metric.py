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

class PrototypeKNNClassifier():
    def __init__(self, num_classes=2, embedding_size=128, device='cpu'):
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.prototypes = torch.zeros(num_classes, embedding_size).to(device)
        return

    def accumulate_prototpyes(self, x, y):
        """
        x: torch.tensor of shape ([embedding_size])
        y: torch.tensor of shape (int)
        """
        if torch.sum(self.prototypes[y]) == 0: # There is no prototype for this class yet
            self.prototypes[y] = x
        else:
            self.prototypes[y] = (self.prototypes[y] + x) / 2
        return

    def reset(self):
        self.prototypes = torch.zeros(self.num_classes, self.embedding_size)

    def predict(self, x):
        """
        x: torch.tensor of shape ([batch_size, embedding_size])
        """
        x = torch.unsqueeze(x, dim=1) # Adding needed dummy dimension
        out = torch.nn.functional.cosine_similarity(x, self.prototypes, dim=2)
        out = torch.argmax(out, dim=1)
        return out


class KNNProbingAccuracyMetric(GenericPluginMetric[float]):
    """
    1. For each class compute the centroid representation from the entire train stream
    2. Use cosine similairty to the class anchors to determine nearest neighbors
    Two options: prototype and full? (full will be super slow in execution)
    """
    def __init__(self, train_stream, eval_all=False, num_classes=10,
        batch_size=32, num_workers=0):
        self._accuracy = Accuracy() # metric calculation container
        super(KNNProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='experience', emit_at='experience',
            mode='eval')

        self.eval_all = eval_all # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_stream = train_stream
        self.num_exps_seen = 0

        self.training_complete = False

        self.knn_classifier = None
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
    
        # TODO: implement prediction of knn for class-incremental
        # TODO: implement prediction of knn for task-incremental
            # (needs one knn classifier per task)

        # Get representation of batch
        x_rep = strategy.model.last_features.detach()

        # Get Prediciton from KNN_Classifier(s)
        out = self.knn_classifier.predict(x_rep)
        
        # Update the accuracy measure    
        self._accuracy.update(out, strategy.mb_y, task_labels)
        return


    def before_eval_exp(self, strategy: 'BaseStrategy'):
        # Check if LinearProbe already trained
        if not self.training_complete:
        
            self.training_complete = True

            # Initialize the train_stream for all seen experiences
            if self.eval_all:
                self.num_exps_seen = len(self.train_stream)-1
            curr_exp_data_stream = self.train_stream[:(self.num_exps_seen+1)]
            curr_exp_data = []
            curr_exp_targets = []
            for exp in curr_exp_data_stream:
                curr_exp_targets.append(torch.unique(torch.tensor(exp.dataset.targets)))
                curr_exp_data.append(exp.dataset)
            num_curr_exp_targets = len(torch.unique(torch.cat(curr_exp_targets)))
            print("\n Current num targets:", num_curr_exp_targets)
            lp_dataset = torch.utils.data.ConcatDataset(curr_exp_data)
            lp_dataloader = torch.utils.data.DataLoader(lp_dataset, batch_size=self.batch_size, 
                        shuffle=False, num_workers=self.num_workers)

            # Initialize KNN Classifier(s)
            print("\nKnnClassifier:", num_curr_exp_targets, strategy.model.classifier.in_features)
            self.knn_classifier = PrototypeKNNClassifier(
                    num_classes=num_curr_exp_targets, 
                    embedding_size=strategy.model.classifier.in_features,
                    device=strategy.device
            )
            
            print("Collected dataset and loader...")
            # Accumulate the prototypes for each class according to ground truth labels
            print("Accumulating prototypes for knn classification")
            for i, mbatch in tqdm(enumerate(lp_dataloader), total=len(lp_dataloader)):
                x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
                x = x.to(strategy.device)
                y = y.to(strategy.device)
                # Get (l2-normalized) representations for current batch
                x_rep = torch.nn.functional.normalize(strategy.model.feature_extractor(x).detach(), dim=1)
                for l in torch.unique(y):
                    #proto_l = torch.sum(x_rep[torch.where(y==l)], dim=0)
                    proto_l = x_rep[torch.where(y==l)].mean(dim=0)
                    #print("\n", proto_l)
                    #print("proto shape", proto_l.shape)
                    # Accumulate prototype for class l
                    self.knn_classifier.accumulate_prototpyes(proto_l, l)

        super().before_eval_exp(strategy)
        if self._reset_at == 'experience' and self._mode == 'eval':
            self.reset(strategy)

        return

    def after_eval(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # Release the lock on LinearProbe training
        self.training_complete = False
        print("\nReleased Flag for Linear Probe Training")
        self.num_exps_seen += 1
        return


    