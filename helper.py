from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
from pathlib import Path
from typing import List
import uuid
import random
import numpy
import torch
import datetime
import argparse
from distutils.util import strtobool
import time

from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from torchvision import transforms

# Avalanche
from avalanche.logging import TextLogger, TensorboardLogger, WandBLogger
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10, RotatedMNIST, PermutedMNIST
from avalanche.evaluation.metrics import ExperienceForgetting, StreamForgetting, accuracy_metrics, loss_metrics, \
    StreamConfusionMatrix, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins import GEMPlugin, AGEMPlugin
from avalanche.training.strategies import Naive
from avalanche.models.dynamic_modules import MultiHeadClassifier
from avalanche.training.strategies.base_strategy import BaseStrategy

# CUSTOM
from src.utils import MetricOverSeed
from src.model import FeatClassifierModel, MLPfeat, ResNet18feat, L2NormalizeLayer
from src.eval.continual_eval import ContinualEvaluationPhasePlugin
from src.eval.continual_eval_metrics import TaskTrackingLossPluginMetric, \
    TaskTrackingGradnormPluginMetric, TaskTrackingFeatureDriftPluginMetric, TaskTrackingAccuracyPluginMetric, \
    TaskAveragingPluginMetric, WindowedForgettingPluginMetric, \
    TaskTrackingMINAccuracyPluginMetric, TrackingLCAPluginMetric, WCACCPluginMetric, WindowedPlasticityPluginMetric
from src.eval.minibatch_logging import StrategyAttributeAdderPlugin, StrategyAttributeTrackerPlugin
from src.utils import ExpLRSchedulerPlugin, IterationsInsteadOfEpochs
from src.benchmarks.domainnet_benchmark import MiniDomainNetBenchmark
from src.benchmarks.digits_benchmark import DigitsBenchmark
from src.methods.lwf_standard import LwFStandard
from src.methods.ewc_standard import EWCStandardPlugin
from src.methods.replay import ERPlugin, ACE_CE_Loss #, ERStrategy
from src.methods.replay_m import ERPluginOrig #, ERStrategyOrig
from src.methods.gem_standard import GEMStandard, GEMStandardPlugin
from src.benchmarks.miniimagenet_benchmark import SplitMiniImageNet
from src.methods.grad_clip import GradClipPlugin
from src.methods.backbone_freeze import FreezeBackbonePlugin
from src.methods.lr_warmup_scheduler import LRWarmupScheduler, LinearWarmup
from src.methods.er_agem import ERAGEMPlugin
from src.methods.reinit_backbone import ReInitBackbonePlugin
from src.methods.linear_probing import LinearProbePlugin
from src.methods.agem_bgd import AGEM_BGD
from src.methods.agem_standard import AGEMPlugin
from src.methods.er_geco import GECOERPlugin
from src.methods.der import DERPlugin

def args_to_tensorboard(writer, args):
    """
    Copyright: OCDVAE
    Takes command line parser arguments and formats them to
    display them in TensorBoard text.
    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        args (dict): dictionary of command line arguments
    """
    txt = ""
    for arg in sorted(vars(args)):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)
    return


def get_scenario(args, seed):
    print(f"\n[SCENARIO] {args.scenario}, Task Incr = {args.task_incr}")

    if args.scenario == 'smnist':  #
        args.input_size = (1, 28, 28)
        n_classes = 10
        n_experiences = 5
        scenario = SplitMNIST(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                              fixed_class_order=[i for i in range(n_classes)])
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    elif args.scenario == 'pmnist':  #
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (1, 28, 28)
        n_classes = 10
        scenario = PermutedMNIST(n_experiences=5, seed=seed)
        scenario.n_classes = n_classes

    elif args.scenario == 'rotmnist':  # Domain-incremental
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (1, 28, 28)
        n_classes = 10
        n_experiences = 20
        scenario = RotatedMNIST(n_experiences=n_experiences,
                                rotations_list=[t * (180 / n_experiences) for t in range(n_experiences)])
        scenario.n_classes = n_classes

    elif args.scenario == 'digits':  # Domain-incremental
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (3, 32, 32)
        n_classes = 10
        scenario = DigitsBenchmark()
        scenario.n_classes = n_classes

    elif args.scenario == 'minidomainnet':
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (3, 96, 96)
        n_classes = 126
        scenario = MiniDomainNetBenchmark(dataset_root=args.dset_rootpath)
        scenario.n_classes = n_classes

    elif args.scenario == 'cifar10':
        args.input_size = (3, 32, 32)
        n_classes = 10
        n_experiences = 5

        # Init minimal transforms
        minimal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        # Init augmentation transforms (horizontal flip + random crop)
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]) 
        train_transform = minimal_transform
        test_transform = minimal_transform

        if args.advanced_data_aug:
            train_transform = aug_transform


        scenario = SplitCIFAR10(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                                fixed_class_order=[i for i in range(n_classes)],
                                train_transform=train_transform,
                                eval_transform=test_transform)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    elif args.scenario == 'miniimgnet':
        args.input_size = (3, 84, 84)
        n_classes = 100
        n_experiences = 20
        if not args.num_experiences is None:
            n_experiences = args.num_experiences
        scenario = SplitMiniImageNet(args.dset_rootpath, n_experiences=n_experiences, return_task_id=args.task_incr,
                                     seed=seed, per_exp_classes=args.per_exp_classes_dict,
                                     fixed_class_order=[i for i in range(n_classes)], preprocessed=True)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head
       
        # exp = scenario.train_stream[0].dataset
        # for exp in scenario.train_stream:
        #     dl = torch.utils.data.DataLoader(exp.dataset, batch_size=3, shuffle=True, num_workers=0)
        #     for data in dl:
        #         print(data)
        #         break
        #     continue    
        # import sys;sys.exit()

    else:
        raise ValueError("Wrong scenario name.")

    # Cutoff if applicable
    scenario.train_stream = scenario.train_stream[: args.partial_num_tasks]
    scenario.test_stream = scenario.test_stream[: args.partial_num_tasks]

    print(f"Scenario = {args.scenario}")
    return scenario


def get_continual_evaluation_plugins(args, scenario):
    """Plugins for per-iteration evaluation in Avalanche."""
    assert args.eval_periodicity >= 1, "Need positive "

    if args.eval_with_test_data:
        args.evalstream_during_training = scenario.test_stream  # Entire test stream
    else:
        args.evalstream_during_training = scenario.train_stream  # Entire train stream
    print(f"Evaluating on stream (eval={args.eval_with_test_data}): {args.evalstream_during_training}")

    # Metrics
    loss_tracking = TaskTrackingLossPluginMetric()
    
    # Expensive metrics
    gradnorm_tracking = None
    if args.track_gradnorm:
        gradnorm_tracking = TaskTrackingGradnormPluginMetric() # if args.track_gradnorm else None  # Memory+compute expensive
    # featdrift_tracking = None
    # if args.track_featdrift:
    #     featdrift_tracking = TaskTrackingFeatureDriftPluginMetric() # if args.track_features else None  # Memory expensive

    # Acc derived plugins
    acc_tracking = TaskTrackingAccuracyPluginMetric()
    #lca = TrackingLCAPluginMetric()

    acc_min = TaskTrackingMINAccuracyPluginMetric()
    acc_min_avg = TaskAveragingPluginMetric(acc_min)
    wc_acc_avg = WCACCPluginMetric(acc_min)

    # wforg_10 = WindowedForgettingPluginMetric(window_size=10)
    # wforg_10_avg = TaskAveragingPluginMetric(wforg_10)
    # wforg_100 = WindowedForgettingPluginMetric(window_size=100)
    # wforg_100_avg = TaskAveragingPluginMetric(wforg_100)

    # wplast_10 = WindowedPlasticityPluginMetric(window_size=10)
    # wplast_10_avg = TaskAveragingPluginMetric(wplast_10)
    # wplast_100 = WindowedPlasticityPluginMetric(window_size=100)
    # wplast_100_avg = TaskAveragingPluginMetric(wplast_100)

    tracking_plugins = [
        loss_tracking, gradnorm_tracking, acc_tracking, #featdrift_tracking
        #lca,  # LCA from A-GEM (is always avged)
        acc_min, acc_min_avg, wc_acc_avg,  # min-acc/worst-case accuracy
        # wforg_10, wforg_10_avg,  # Per-task metric, than avging metric
        # wforg_100, wforg_100_avg,  # Per-task metric, than avging metric
        # wplast_10, wplast_10_avg,  # Per-task metric, than avging metric
        # wplast_100, wplast_100_avg,  # Per-task metric, than avging metric
    ]
    tracking_plugins = [p for p in tracking_plugins if p is not None]

    trackerphase_plugin = ContinualEvaluationPhasePlugin(tracking_plugins=tracking_plugins,
                                                         max_task_subset_size=args.eval_task_subset_size,
                                                         eval_stream=args.evalstream_during_training,
                                                         eval_max_iterations=args.eval_max_iterations,
                                                         mb_update_freq=args.eval_periodicity,
                                                         num_workers=args.num_workers,
                                                         )
    return [trackerphase_plugin, *tracking_plugins]


def get_metrics(scenario, args):
    """Metrics are calculated efficiently as running avgs."""

    # Pass plugins, but stat_collector must be called first
    minibatch_tracker_plugins = []

    # Stats on external tracking stream
    if args.enable_continual_eval:
        tracking_plugins = get_continual_evaluation_plugins(args, scenario)
        minibatch_tracker_plugins.extend(tracking_plugins)

    # Current minibatch stats per class
    # if args.track_class_stats:
    #     for y in range(scenario.n_classes):
    #         minibatch_tracker_plugins.extend([
    #             # Loss components
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_numerator_c{y}"]),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_denominator_c{y}"]),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_c{y}"]),

    #             # Prototypes
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protodelta_weight_c{y}']),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protonorm_weight_c{y}']),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protodelta_bias_c{y}']),
    #             StrategyAttributeTrackerPlugin(strategy_attr=[f'protonorm_bias_c{y}']),
    #         ])

    # METRICS FOR STRATEGIES (Will only track if available for method)
    # minibatch_tracker_plugins.extend([
    #     StrategyAttributeTrackerPlugin(strategy_attr=["loss_new"]),
    #     StrategyAttributeTrackerPlugin(strategy_attr=["loss_reg"]),
    #     StrategyAttributeTrackerPlugin(strategy_attr=["gradnorm_stab"]),
    #     StrategyAttributeTrackerPlugin(strategy_attr=["avg_gradnorm_G"]),
    # ])

    metrics = [
        accuracy_metrics(experience=True, stream=True),
        #loss_metrics(minibatch=True, experience=True, stream=True),
        #ExperienceForgetting(),  # Test only
        #StreamForgetting(),  # Test only
        #StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=True),

        # CONTINUAL EVAL
        *minibatch_tracker_plugins,

        # LOG OTHER STATS
        #timing_metrics(epoch=True, experience=False),
        # cpu_usage_metrics(experience=True),
        # DiskUsageMonitor(),
        # MinibatchMaxRAM(),
        # GpuUsageMonitor(0),
    ]
    return metrics

def get_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        raise ValueError()
    return optimizer

def get_strategy(args, model, eval_plugin, scenario, device, plugins=None):
    plugins = [] if plugins is None else plugins

    # CRIT/OPTIM
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)

    # lr-schedule over experiences
    # if args.lr_milestones is not None:
    #     assert args.lr_decay is not None, "Should specify lr_decay when specifying lr_milestones"
    #     milestones = [int(m) for m in args.lr_milestones.split(',')]
    #     sched = ExpLRSchedulerPlugin(MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay))
    #     plugins.append(sched)
    #     print(f"MultiStepLR schedule over experiences, decaying '{args.lr_decay}' at exps '{milestones}'")

    # Use Iterations if defined
    if args.iterations_per_task is not None:
        args.epochs = int(1e9)
        it_stopper = IterationsInsteadOfEpochs(max_iterations=args.iterations_per_task)
        plugins.append(it_stopper)


    # STRATEGY
    if args.strategy == 'finetune':
        strategy = Naive(model, optimizer, criterion,
                        train_epochs=args.epochs, device=device,
                        train_mb_size=args.bs, evaluator=eval_plugin,
                        plugins=plugins
                        )

    elif args.strategy == 'ER_avl':
        print("\n Using ER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[ReplayPlugin(mem_size=args.mem_size)])
        
    elif args.strategy == 'ER':
        print("\n Using ER_custom strategy")
        # if args.ace_ce_loss: 
        #         criterion = ACE_CE_Loss(device=device)
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[ERPlugin(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences, device=device,
                lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup, do_decay_lmbda=args.do_decay_lmbda,
                ace_ce_loss=args.ace_ce_loss)]
        ) 

    elif args.strategy == 'ER_GECO':
        print("\n Using ER_GECO strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[GECOERPlugin(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences, device=device,
            alpha=0.99, lagrange_update_set=1)]
        ) 

    elif args.strategy == 'ER_orig':
        print("\n Using ER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[ERPluginOrig(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences,)]
        )

    elif args.strategy == 'DER':
        print("\n Using DER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[DERPlugin(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences, device=device,
                lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup, do_decay_lmbda=args.do_decay_lmbda,
                ace_ce_loss=args.ace_ce_loss)]
        )


    elif args.strategy == 'EWC':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
            mode='online', keep_importance_data=False)]
        )
    
    elif args.strategy == 'EWC_separate':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
            mode='separate', keep_importance_data=False)]
        )
    
    elif args.strategy == 'EWC_AGEM':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
                        mode='online', keep_importance_data=False),
                    AGEMPlugin(patterns_per_experience=args.mem_size//scenario.n_experiences,
                        sample_size=args.sample_size)]
        )
    
    elif args.strategy == 'EWC_AGEM_separate':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
                        mode='separate', keep_importance_data=False),
                    AGEMPlugin(patterns_per_experience=args.mem_size//scenario.n_experiences,
                        sample_size=args.sample_size)]
        )


    elif args.strategy == 'GEM':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[GEMStandardPlugin(args.mem_size // scenario.n_experiences, args.gem_gamma)])

    elif args.strategy == 'AGEM_avl':
        print("\n Using AGEM_avl strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[AGEMPlugin(patterns_per_experience=args.mem_size//scenario.n_experiences,
                        sample_size=args.mem_size//scenario.n_experiences)])

    elif args.strategy == 'AGEM':
        print("\n Using AGEM strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[AGEMPlugin(n_total_memories=args.mem_size, sample_size=args.sample_size,
                lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup)])

    elif args.strategy == 'ER_AGEM':
        print("\n Using ER_AGEM_custom strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[ERAGEMPlugin(n_total_memories=args.mem_size, sample_size=args.sample_size,
            lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup, do_decay_lmbda=args.do_decay_lmbda)])

    elif args.strategy == 'AGEM_BGD':
        print("\nUsing AGEM_BGD strategy")
        strategy = AGEM_BGD(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=args.epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device, 
        plugins=[])

    else:
        raise NotImplementedError(f"Non existing strategy arg: {args.strategy}")

    # Additional auxiliary plugins
    if not args.grad_clip is None:
        add_plugin = GradClipPlugin(clip_value=args.grad_clip)
        strategy.plugins.append(add_plugin)
    if args.freeze_backbone:
        strategy.plugins.append(FreezeBackbonePlugin())

    # LRScheduler (warmup)
    if args.lr_warmup_steps > 0:
        lr_plugin = LRWarmupScheduler(
                LinearWarmup(
                    optimizer,
                    total_iters=args.lr_warmup_steps,
                    start_factor=0.0,
                    end_factor=1.0
                ), 
            ) # reset_scheduler=True, reset_lr=True, step_granularity='iteration',
            # first_epoch_only=False, first_exp_only=False, verbose=True)
        strategy.plugins.append(lr_plugin)
    
    # Re-Initialize Model (Only for an experiment concerning (A)GEM)
    if args.reinit_model:
        reinit_plugin = ReInitBackbonePlugin()
        strategy.plugins.append(reinit_plugin)
        print("Added re-init plugin!")
    
    if args.linear_probing > 0:
        strategy.plugins.append(
            LinearProbePlugin(num_epochs=args.linear_probing)
        )
        print("Added linear probing plugin!")

    print(f"Running strategy:{strategy}")
    if hasattr(strategy, 'plugins'):
        print(f"with Plugins: {strategy.plugins}")
    return strategy


def overwrite_args_with_config(args):
    """
    Directly overwrite the input args with values defined in config yaml file.
    Only if args.config_path is defined.
    """
    if args.config_path is None:
        return
    assert os.path.isfile(args.config_path), f"Config file does not exist: {args.config_path}"

    import yaml
    with open(args.config_path, 'r') as stream:
        arg_configs = yaml.safe_load(stream)

    for arg_name, arg_val in arg_configs.items():  # Overwrite
        assert hasattr(args, arg_name), \
            f"'{arg_name}' defined in config is not specified in args, config: {args.config_path}"
        if isinstance(arg_val, (list, tuple)):
            arg_val = arg_val[0]  # unpack first
        setattr(args, arg_name, arg_val)
    print(f"Overriden args with config: {args.config_path}")