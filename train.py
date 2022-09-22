from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



print("Loading libs...")
from avalanche.training.strategies.base_strategy import BaseStrategy
import os
import math
from pathlib import Path
from typing import List
import uuid
import random
import numpy
import torch
from datetime import datetime
import argparse
from distutils.util import strtobool
import time

from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

# Avalanche
import avalanche as avl
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# CUSTOM
from src.utils import MetricOverSeed
from src.model import get_model
from src.eval.minibatch_logging import StrategyAttributeAdderPlugin, StrategyAttributeTrackerPlugin

import helper
from cmd_parser import get_arg_parser
print("loading libs done..")


'''
Init device
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Init Argparser
'''
parser = get_arg_parser()
# Load args from commandline
args = parser.parse_args()



# Override args from yaml file
helper.overwrite_args_with_config(args)

# Process potentially passed dict
if not args.per_exp_classes_dict is None:
    print("Converting per_exp_classes_dict...")
    print(args.per_exp_classes_dict)
    tmp_dict = {}
    for key, value in enumerate(args.per_exp_classes_dict):
        tmp_dict[key] = int(value)
    args.per_exp_classes_dict = tmp_dict
    print(args.per_exp_classes_dict)
    print("done...\n")


'''
Setup results directories
'''
# args.now = str(datetime.datetime.now().date()) + "_" + '-'.join(str(datetime.datetime.now()strftime("%Y.%m.%d-%H:%M")).split(':')[:-1])
# args.uid = uuid.uuid4().hex
# args.exp_name = '_'.join([args.exp_name, f"now={args.now}", f"uid={args.uid}"])
# print(f"STARTING TIME: {args.now}\nEXP NAME:{args.exp_name}\nargs: {vars(args)}")

args.now = datetime.now().strftime("%Y.%m.%d-%H:%M")

# Paths
optim_str = "no_reset_optim"
if args.reset_optim_each_exp:
    optim_str = "reset_optim"
args.setupname = '_'.join([args.exp_name, args.strategy, args.backbone, str(args.lmbda), args.scenario, args.optim, optim_str, f"e={args.epochs[0]}", args.now])
args.results_path = Path(os.path.join(args.save_path, args.setupname)).resolve()
args.eval_results_dir = args.results_path / 'results_summary'  # Eval results
for path in [args.results_path, args.eval_results_dir]:
    path.mkdir(parents=True, exist_ok=True)


'''
Create Scenario
'''
scenario, train_transform = helper.get_scenario(args, seed=args.seed)

'''
Create Logger
'''
loggers = []
# Tensorboard
#tb_log_dir = os.path.join(args.results_path, 'tb_run', f'seed={args.seed}')  # Group all runs
tb_log_dir = os.path.join(args.results_path)  # Group all runs
tb_logger = TensorboardLogger(tb_log_dir=tb_log_dir)
loggers.append(tb_logger)  # log to Tensorboard
print(f"[Tensorboard] tb_log_dir={tb_log_dir}")
# Terminal
print_logger = TextLogger() 
if args.disable_pbar:
    print_logger = InteractiveLogger()  # print to stdout
loggers.append(print_logger)

'''
Init Model
'''
model = get_model(args, scenario.n_classes)


'''
Init Evaluation
'''
metrics = helper.get_metrics(scenario, args)
eval_plugin = EvaluationPlugin(*metrics, loggers=loggers, benchmark=scenario)


'''
Init Strategy
'''
strategy_plugins = [StrategyAttributeAdderPlugin(list(range(scenario.n_classes)))]
strategy = helper.get_strategy(args, model, eval_plugin, scenario, device, 
                plugins=strategy_plugins, train_transform=train_transform)

'''
Store args to tensorboard
'''
helper.args_to_tensorboard(tb_logger.writer, args)


'''
Train Loop
'''
print('Starting experiment...')
for experience in scenario.train_stream:
    # TRAIN
    print(f"\n{'-' * 40} TRAIN {'-' * 40}")
    print(f"Start training on experience {experience.current_experience}")
    strategy.train(experience, num_workers=args.num_workers, eval_streams=None)
    print(f"End training on experience {experience.current_experience}")

    # EVAL ALL TASKS (ON TASK TRANSITION)
    print(f"\n{'=' * 40} EVAL {'=' * 40}")
    print(f'Standard Continual Learning eval on entire test set on task transition.')
    task_results_file = args.eval_results_dir / f'seed={args.seed}' / f'task{experience.current_experience}_results.pt'
    task_results_file.parent.mkdir(parents=True, exist_ok=True)
    res = strategy.eval(scenario.test_stream)  # Gathered by EvalLogger

    # Store eval task results
    task_metrics = dict(strategy.evaluator.all_metric_results)
    torch.save(task_metrics, task_results_file)
    print(f"[FILE:TASK-RESULTS]: {task_results_file}")

    # Reset optimizer
    if args.reset_optim_each_exp:
        strategy.optimizer = helper.get_optimizer(args, strategy.model)
        print("\nRESET OPTIMIZER")
