from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from typing import List
import torch

from torchvision import transforms

# Avalanche
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10, RotatedMNIST, PermutedMNIST
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.evaluation.metrics import ExperienceForgetting, StreamForgetting, accuracy_metrics, loss_metrics, \
    StreamConfusionMatrix, timing_metrics
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins import GEMPlugin, AGEMPlugin
from avalanche.training.strategies import Naive
from avalanche.models.dynamic_modules import MultiHeadClassifier
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.benchmarks.datasets import default_dataset_location

# CUSTOM
from src.eval.continual_eval import ContinualEvaluationPhasePlugin
from src.eval.continual_eval_metrics import TaskTrackingLossPluginMetric, \
    TaskTrackingGradnormPluginMetric, TaskTrackingFeatureDriftPluginMetric, TaskTrackingAccuracyPluginMetric, \
    TaskAveragingPluginMetric, WindowedForgettingPluginMetric, \
    TaskTrackingMINAccuracyPluginMetric, TrackingLCAPluginMetric, WCACCPluginMetric, WindowedPlasticityPluginMetric
from src.eval.linear_probing_metric import LinearProbingAccuracyMetric
from src.eval.knn_probing_metric import KNNProbingAccuracyMetric
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
from src.methods.agem_bgd import AGEM_BGD
from src.methods.agem_standard import AGEMPlugin
from src.methods.er_geco import GECOERPlugin
from src.methods.der import DERPlugin
from src.methods.epoch_adapter import EpochLengthAdapterPlugin
from src.methods.store_models import StoreModelsPlugin

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

def get_condor_dataset_root(args, dset_name):
    # If the scratch_dir is not used, return the original dataset root
    if not args.use_condor_sc_dir:
        return args.dset_rootpath

    # Get the original dataset source
    orig_dset_root = args.dset_rootpath
    if args.dset_rootpath is None:
        orig_dset_root = default_dataset_location(dset_name)
    print("orig_dset_root: {}".format(orig_dset_root))
    # Get the assigned scratch_dir location
    condor_scratch_dir = os.environ.get('_CONDOR_SCRATCH_DIR', None)
    print("scratch_dir is:", condor_scratch_dir)

    # Define the dataset root on scratch dir
    print("dset_name: {}".format(dset_name))
    new_dset_root = os.path.join(condor_scratch_dir, dset_name)
    print("new dset root on scratch dir:", new_dset_root)
    print(condor_scratch_dir + "/" + dset_name)

    # Check if the dataset was alreday downloaded in original root and transfers if true
    if not orig_dset_root is None:
        if os.path.exists(orig_dset_root):
            if not os.path.exists(new_dset_root):    
                print("Copying existing dataset to condor scratch dir...")
                shutil.copytree(orig_dset_root, new_dset_root)
                print("done...")
            else:
                print("Dataset already exists in condor scratch dir")
                print("Do nothing...")
    else:
        print("No original dataset root! Please specify a dataset rootpath if avalanche automatic download fails...") 
    return new_dset_root


def get_scenario(args, seed):
    print(f"\n[SCENARIO] {args.scenario}, Task Incr = {args.task_incr}")

    # Prepare general transforms
    train_transform = None
    test_transform = None

    if args.scenario in ["smnist", "pmnist", "rotmnist"]:
        args.input_size = (1, 28, 28)
    elif args.scenario in ["cifar10", "cifar100", "digits"]:
        args.input_size = (3, 32, 32)
    elif args.scenario in ["minidomainnet"]:
        args.input_size = (3, 96, 96)
    elif args.scenario in ["miniimgnet"]:
        args.input_size = (3, 84, 84)
    
    if not args.overwrite_input_size is None:
        args.input_size = (args.input_size[0], args.overwrite_input_size[0], args.overwrite_input_size[1])

    to_pil = transforms.Compose([transforms.ToPILImage()])
    resize = transforms.Compose([transforms.Resize(size=(args.input_size[1], args.input_size[2]), interpolation=transforms.InterpolationMode.NEAREST)])
    sim_clr = transforms.Compose([
        transforms.RandomResizedCrop(size=(args.input_size[1], args.input_size[2])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, # NOTE: SimCLR uses 0.8
                                    contrast=0.5, # NOTE: SimCLR uses 0.8
                                    saturation=0.5, # NOTE: SimCLR uses 0.8
                                    hue=0.1) # NOTE: SimCLR uses 0.2
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
    ])
    to_tensor = transforms.Compose([transforms.ToTensor()])

    # Prepare datasets/scenarios
    if args.scenario == 'smnist':  #
        n_classes = 10
        n_experiences = 5
        new_dset_root = get_condor_dataset_root(args, "mnist")
        scenario = SplitMNIST(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                              fixed_class_order=[i for i in range(n_classes)], dataset_root=new_dset_root)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    elif args.scenario == 'pmnist':  #
        assert not args.task_incr, "Domain incremental can't be multi-head."
        n_classes = 10
        scenario = PermutedMNIST(n_experiences=5, seed=seed)
        scenario.n_classes = n_classes

    elif args.scenario == 'rotmnist':  # Domain-incremental
        assert not args.task_incr, "Domain incremental can't be multi-head."
        n_classes = 10
        n_experiences = 20
        scenario = RotatedMNIST(n_experiences=n_experiences,
                                rotations_list=[t * (180 / n_experiences) for t in range(n_experiences)])
        scenario.n_classes = n_classes

    elif args.scenario == 'digits':  # Domain-incremental
        assert not args.task_incr, "Domain incremental can't be multi-head."
        n_classes = 10
        scenario = DigitsBenchmark()
        scenario.n_classes = n_classes

    elif args.scenario == 'minidomainnet':
        assert not args.task_incr, "Domain incremental can't be multi-head."
        n_classes = 126
        scenario = MiniDomainNetBenchmark(dataset_root=args.dset_rootpath)
        scenario.n_classes = n_classes

    # CIFAR10
    elif args.scenario == 'cifar10':
        n_classes = 10
        n_experiences = 5
 
        # Compose transforms
        normalize = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        minimal_transform = transforms.Compose([resize, to_tensor, normalize])
        train_transform = minimal_transform
        test_transform = minimal_transform
        if args.use_simclr_aug:
            print("Using SimCLR Augmentation")
            train_transform = transforms.Compose([resize, sim_clr, to_tensor, normalize])
            
        new_dset_root = get_condor_dataset_root(args, "cifar10")
        scenario = SplitCIFAR10(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                                fixed_class_order=[i for i in range(n_classes)],
                                train_transform=train_transform,
                                eval_transform=test_transform,
                                dataset_root=new_dset_root)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head
 
    # CIFAR100
    elif args.scenario == 'cifar100':
        n_classes = 100
        n_experiences = 10
        if not args.num_experiences is None:
            n_experiences = args.num_experiences

        
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        if "_pt" in args.backbone:
            normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        minimal_transform = transforms.Compose([resize, to_tensor, normalize])       
        train_transform = minimal_transform
        test_transform = minimal_transform

        if args.use_simclr_aug:
            print("Using SimCLR Augmentation")
            train_transform = transforms.Compose([resize, sim_clr, to_tensor, normalize])

        new_dset_root = get_condor_dataset_root(args, "cifar100")
        scenario = SplitCIFAR100(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                                fixed_class_order=[i for i in range(n_classes)],
                                train_transform=train_transform,
                                eval_transform=test_transform,
                                dataset_root=new_dset_root)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences

    # MiniImageNet
    elif args.scenario == 'miniimgnet':
        n_classes = 100
        n_experiences = 20
        if not args.num_experiences is None:
            n_experiences = args.num_experiences

        normalize = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if "_pt" in args.backbone:
            normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        minimal_transform = transforms.Compose([to_pil, resize, to_tensor, normalize])
        train_transform = minimal_transform
        test_transform = minimal_transform
        if args.use_simclr_aug:
            print("Using SimCLR Augmentation")  
            train_transform = transforms.Compose([to_pil, resize, sim_clr, to_tensor, normalize])

        new_dset_root = get_condor_dataset_root(args, dset_name="miniimgnet")    
        scenario = SplitMiniImageNet(new_dset_root, n_experiences=n_experiences, return_task_id=args.task_incr, # NOTE: args.dset_rootpath as first argument (original code)
                                     seed=seed, per_exp_classes=args.per_exp_classes_dict,
                                     fixed_class_order=[i for i in range(n_classes)], preprocessed=True,
                                     train_transform=train_transform, test_transform=test_transform)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    else:
        raise ValueError("Wrong scenario name.")

    # Cutoff if applicable
    scenario.train_stream = scenario.train_stream[: args.partial_num_tasks]
    scenario.test_stream = scenario.test_stream[: args.partial_num_tasks]

    print(f"Scenario = {args.scenario}")
    return scenario, train_transform


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

    # Linear Probing Evaluation
    if args.use_lp_eval:
        print("\nAdding a probing eval plugin")
        if args.use_lp_eval == "linear":
            print("Using linear probe")
            metrics.append(LinearProbingAccuracyMetric(train_stream=scenario.train_stream, test_stream=scenario.test_stream,
                eval_all=args.lp_eval_all, force_task_eval=args.lp_force_task_eval,
                num_finetune_epochs=args.lp_finetune_epochs,
                skip_initial_eval=args.skip_initial_eval)
            )
        elif args.use_lp_eval == "knn":
            print("Using knn probe")
            metrics.append(KNNProbingAccuracyMetric(train_stream=scenario.train_stream, eval_all=args.lp_eval_all,
                num_classes=scenario.n_classes)
            )
    return metrics

def get_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        raise ValueError()
    return optimizer

def get_strategy(args, model, eval_plugin, scenario, device, 
            plugins=None, train_transform=None):
    plugins = [] if plugins is None else plugins

    # CRIT/OPTIM
    criterion = torch.nn.CrossEntropyLoss()
    if args.ace_ce_loss:
        criterion = ACE_CE_Loss(device=device)
        print("\nUsing ACE_CE_Loss")
    optimizer = get_optimizer(args, model)

    initial_epochs = args.epochs[0]

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
                        train_epochs=initial_epochs, device=device,
                        train_mb_size=args.bs, evaluator=eval_plugin,
                        plugins=plugins
                        )

    elif args.strategy == 'ER_avl':
        print("\n Using ER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[ReplayPlugin(mem_size=args.mem_size)])
        
    elif args.strategy == 'ER':
        print("\n Using ER_custom strategy")
        # if args.ace_ce_loss: 
        #         criterion = ACE_CE_Loss(device=device)
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[ERPlugin(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences, device=device,
                lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup, do_decay_lmbda=args.do_decay_lmbda,
                ace_ce_loss=args.ace_ce_loss)]
        ) 

    elif args.strategy == 'ER_GECO':
        print("\n Using ER_GECO strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[GECOERPlugin(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences, device=device,
            alpha=0.99, lagrange_update_step=1)]
        ) 

    elif args.strategy == 'ER_orig':
        print("\n Using ER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[ERPluginOrig(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences,)]
        )

    elif args.strategy == 'DER':
        print("\n Using DER strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[
                DERPlugin(n_total_memories=args.mem_size, num_tasks=scenario.n_experiences, device=device,
                    lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup, do_decay_lmbda=args.do_decay_lmbda,
                    ace_ce_loss=args.ace_ce_loss,
                    train_transform=train_transform
                )
            ]
        )


    elif args.strategy == 'EWC':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
            mode='online', keep_importance_data=False)]
        )
    
    elif args.strategy == 'EWC_separate':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
            mode='separate', keep_importance_data=False)]
        )
    
    elif args.strategy == 'EWC_AGEM':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
                        mode='online', keep_importance_data=False),
                    AGEMPlugin(patterns_per_experience=args.mem_size//scenario.n_experiences,
                        sample_size=args.sample_size)]
        )
    
    elif args.strategy == 'EWC_AGEM_separate':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
            train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
            evaluator=eval_plugin, device=device,
            plugins=[EWCStandardPlugin(iw_strength=args.lmbda, 
                        mode='separate', keep_importance_data=False),
                    AGEMPlugin(patterns_per_experience=args.mem_size//scenario.n_experiences,
                        sample_size=args.sample_size)]
        )


    elif args.strategy == 'GEM':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[GEMStandardPlugin(args.mem_size // scenario.n_experiences, args.gem_gamma)])

    elif args.strategy == 'AGEM_avl':
        print("\n Using AGEM_avl strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[AGEMPlugin(patterns_per_experience=args.mem_size//scenario.n_experiences,
                        sample_size=args.mem_size//scenario.n_experiences)])

    elif args.strategy == 'AGEM':
        print("\n Using AGEM strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[AGEMPlugin(n_total_memories=args.mem_size, sample_size=args.sample_size,
                lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup)])

    elif args.strategy == 'ER_AGEM':
        print("\n Using ER_AGEM_custom strategy")
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device,
        plugins=[ERAGEMPlugin(n_total_memories=args.mem_size, sample_size=args.sample_size,
            lmbda=args.lmbda, lmbda_warmup_steps=args.lmbda_warmup, do_decay_lmbda=args.do_decay_lmbda)])

    elif args.strategy == 'AGEM_BGD':
        print("\nUsing AGEM_BGD strategy")
        strategy = AGEM_BGD(model, optimizer, criterion, train_mb_size=args.bs,
        train_epochs=initial_epochs, eval_mb_size=256, eval_every=-1, 
        evaluator=eval_plugin, device=device, 
        plugins=[])

    else:
        raise NotImplementedError(f"Non existing strategy arg: {args.strategy}")
    
    #################
    # Additional auxiliary plugins
    #################
    # Clip Gradients
    if not args.grad_clip is None:
        add_plugin = GradClipPlugin(clip_value=args.grad_clip)
        strategy.plugins.append(add_plugin)
    
    # Freeze backbone
    if args.freeze_backbone:
        strategy.plugins.append(FreezeBackbonePlugin(
            exp_to_freeze_on=args.freeze_after_exp,
            freeze_up_to_layer_name=args.freeze_up_to)
        )

    # Dynamic epoch length adapter
    print("\nNum epochs:", len(args.epochs))
    if len(args.epochs) > 1:
        strategy.plugins.append(
            EpochLengthAdapterPlugin(args.epochs)
        )
        print("Added EpochLengthAdapter!")

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
    
    # Model Storing to Disk
    if args.store_models:
        strategy.plugins.append(StoreModelsPlugin(model_name=args.backbone, model_store_path=args.results_path))

    # Re-Initialize Model (Only for an experiment concerning (A)GEM)
    if args.reinit_model:
        reinit_until_exp = args.reinit_up_to_exp if (not args.reinit_up_to_exp is None) else (scenario.n_experiences+1)
        reinit_plugin = ReInitBackbonePlugin(
                exp_to_reinit_on=args.reinit_after_exp, 
                reinit_until_exp=reinit_until_exp,
                reinit_after_layer_name=args.reinit_layers_after,
                freeze=args.reinit_freeze
            )
        strategy.plugins.append(reinit_plugin)
        print("Added re-init plugin!")

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