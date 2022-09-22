import argparse
from distutils.util import strtobool
import json

def get_arg_parser():
    parser = argparse.ArgumentParser()

    # Meta hyperparams
    parser.add_argument('--exp_name', default="", type=str, help='Name for the experiment.')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Yaml file with config for the args.')

    parser.add_argument('--exp_postfix', type=str, default='#now,#uid',
                        help='Extension of the experiment name. A static name enables continuing if checkpointing is define'
                            'Needed for fixes/gridsearches without needing to construct a whole different directory.'
                            'To use argument values: use # before the term, and for multiple separate with commas.'
                            'e.g. #cuda,#featsize,#now,#uid')
    parser.add_argument('--save_path', type=str, default='./results/', help='save eval results.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for the dataloaders.')
    parser.add_argument('--disable_pbar', default=True, type=lambda x: bool(strtobool(x)), help='Disable progress bar')
    parser.add_argument('--debug', default=False, type=lambda x: bool(strtobool(x)), help='Eval on few samples.')
    parser.add_argument('--n_seeds', default=5, type=int, help='Nb of seeds to run.')
    parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
    parser.add_argument('--deterministic', default=False, type=lambda x: bool(strtobool(x)),
                        help='Set deterministic option for CUDNN backend.')
    parser.add_argument('--wandb', default=False, type=lambda x: bool(strtobool(x)), help="Use wandb for exp tracking.")

    # Dataset
    parser.add_argument('--scenario', type=str, default='smnist',
                        choices=['smnist', 'cifar10', 'cifar100', 'miniimgnet', 'minidomainnet', 'pmnist', 'rotmnist', 'digits'])
    parser.add_argument('--dset_rootpath', default=None, type=str, # NOTE: default='./data' (original code)
                        help='Root path of the downloaded dataset for e.g. Mini-Imagenet')  # Mini Imagenet
    parser.add_argument('--partial_num_tasks', type=int, default=None,
                        help='Up to which task to include, e.g. to consider only first 2 tasks of 5-task Split-MNIST')
    parser.add_argument('--num_experiences', type=int, default=None, 
                        help='Number of experiences to use in the scenario.')
    parser.add_argument('--per_exp_classes_dict', nargs='+', default=None, 
                        help='Dict of per-experience classes to control non uniform distribution of classes per task.')
    parser.add_argument('--use_condor_sc_dir', action='store_true', default=False, help='Whether to use condor scratch dir.')

    # Feature extractor
    parser.add_argument('--featsize', type=int, default=400,
                        help='The feature size output of the feature extractor.'
                            'The classifier uses this embedding as input.')
    parser.add_argument('--backbone', type=str, choices=['input', 'mlp', 'resnet18', 'resnet18_big', 'wrn', 'cifar_mlp', 'simple_cnn', 'vgg11'], default='mlp')
    parser.add_argument('--show_backbone_param_names', action='store_true', default=False, help='Show parameter names of the backbone.')
    parser.add_argument('--use_GAP', default=True, type=lambda x: bool(strtobool(x)),
                        help="Use Global Avg Pooling after feature extractor (for Resnet18).")
    parser.add_argument('--use_maxpool', action='store_true', default=False, help="Use maxpool after feature extractor (for WRN).")
    parser.add_argument('--wrn_depth', type=int, default=14, help="Depth of the wide residual network.")
    parser.add_argument('--wrn_widen_factor', type=int, default=10, help="Widen factor of the wide residual network.")
    parser.add_argument('--wrn_embedding_size', type=int, default=48, help="Embedding size of the wide residual network.")

    # Classifier
    parser.add_argument('--classifier', type=str, choices=['linear', 'norm_embed', 'identity'], default='linear',
                        help='linear classifier (prototype=weight vector for a class)'
                            'For feature-space classifiers, we output the embedding (identity) '
                            'or normalized embedding (norm_embed)')
    parser.add_argument('--lin_bias', default=True, type=lambda x: bool(strtobool(x)),
                        help="Use bias in Linear classifier")

    # Optimization
    parser.add_argument('--optim', type=str, choices=['sgd', 'adam', 'adamW'], default='sgd')
    parser.add_argument('--reset_optim_each_exp', action='store_true', default=False, help='Reset optimizer each exp.')
    parser.add_argument('--bs', type=int, default=128, help='Minibatch size.')
    #parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs/step.')
    parser.add_argument('--epochs', type=int, nargs='+', default=[10], help='Number of epochs per experience. \
        If len(epochs) != n_experiences, then the last epoch is used for the remaining experiences.')
    parser.add_argument('--iterations_per_task', type=int, default=None,
                        help='When this is defined, it overwrites the epochs per task.'
                            'This enables equal compute per task for imbalanced scenarios.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--lr_milestones', type=str, default=None, help='Learning rate epoch decay milestones.')
    parser.add_argument('--lr_decay', type=float, default=None, help='Multiply factor on milestones.')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Number of warmup steps.')

    #parser.add_argument('--advanced_data_aug', action='store_true', default=False, help='Use advanced data augmentation.')
    parser.add_argument('--use_simclr_aug', action='store_true', default=False, help='Use SimCLR data augmentation.')

    # Continual Evaluation
    parser.add_argument('--eval_with_test_data', default=True, type=lambda x: bool(strtobool(x)),
                        help="Continual eval with the test or train stream, default True for test data of datasets.")
    parser.add_argument('--enable_continual_eval', default=True, type=lambda x: bool(strtobool(x)),
                        help='Enable evaluation each eval_periodicity iterations.')
    parser.add_argument('--eval_periodicity', type=int, default=1,
                        help='Periodicity in number of iterations for continual evaluation. (None for no continual eval)')
    parser.add_argument('--eval_task_subset_size', type=int, default=1000,
                        help='Max nb of samples per evaluation task. (-1 if not applicable)')
    parser.add_argument('--eval_max_iterations', type=int, default=-1, help='Max nb of iterations for continual eval.\
                        After this number of iters is reached, no more continual eval is performed. Default value \
                        of -1 means no limit.')

    # Expensive additional continual logging
    parser.add_argument('--track_class_stats', default=False, type=lambda x: bool(strtobool(x)),
                        help="To track per-class prototype statistics, if too many classes might be better to turn off.")
    parser.add_argument('--track_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the gradnorm of the evaluation tasks."
                            "This accumulates computational graphs from the entire task and is very expensive memory wise."
                            "Can be made more feasible with reducing 'eval_task_subset_size'.")
    parser.add_argument('--track_features', default=False, type=lambda x: bool(strtobool(x)),
                        help="Track the features before and after a single update. This is very expensive as "
                            "entire evaluation task dataset features are forwarded and stored twice in memory."
                            "Can be made more feasible with reducing 'eval_task_subset_size'.")
    parser.add_argument('--reduced_tracking', action='store_true', default=False, help='Use reduced tracking metrics.')
    #parser.add_argument('--use_lp_eval', action='store_true', default=False, help='Use Linear Probing in evaluation.')
    parser.add_argument('--use_lp_eval', type=str, default=None, choices=['linear', 'knn'], help='Usa a probing evaluation metric')
    parser.add_argument('--lp_optim', type=str, choices=['sgd', 'adamW'], default='sgd', help='Optimizer for linear probing.')
    parser.add_argument('--lp_lr', type=float, default=1e-3, help='Learning rate for linear probing.')
    parser.add_argument('--lp_eval_all', action='store_true', default=False, help='Use all tasks, always, for Linear Probing evaluation.')
    parser.add_argument('--lp_finetune_epochs', type=int, default=10, help='Number of epochs to finetune Linear Probing.')
    parser.add_argument('--lp_force_task_eval', action='store_true', default=False, help='Force SEPARATE evaluation of all tasks in Linear Probing.')


    # Strategy
    parser.add_argument('--strategy', type=str, default='finetune',
                        choices=['ER_avl', 'ER', 'ER_orig', 'ER_GECO', 'DER',
                                'GEM', 'AGEM_avl', 'AGEM', 'ER_AGEM', 'AGEM_BGD',
                                'EWC', 'EWC_AGEM', 'EWC_separate', 'EWC_AGEM_separate',
                                'finetune', # 'EWC_custom', 'LWF_custom',
                                ])
    parser.add_argument('--task_incr', action='store_true', default=False,
                        help="Give task ids during training to single out the head to the current task.")


    # ER
    parser.add_argument('--Lw_new', type=float, default=0.5,
                        help='Weight for the CE loss on the new data, in range [0,1]')
    parser.add_argument('--record_stability_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                        help="Record the gradnorm of the memory samples in current batch?")
    parser.add_argument('--mem_size', default=1000, type=int, help='Total nb of samples in rehearsal memory.')
    parser.add_argument('--ace_ce_loss', action='store_true', default=False, help='Use ACE_CE loss for ER.')

    # GEM
    parser.add_argument('--gem_gamma', default=0.5, type=float, help='Gem param to favor BWT')

    # LWF
    parser.add_argument('--lwf_alpha', type=float, default=1, help='Distillation loss weight')
    parser.add_argument('--lwf_softmax_t', type=float, default=2, help='Softmax temperature (division).')

    # GradClipping
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping.')
    # BackboneFreezing
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='Freeze backbone.')
    parser.add_argument('--freeze_after_exp', type=int, default=0, help='Freeze backbone after experience n.')
    parser.add_argument('--freeze_up_to', type=str, default=None, help='Freeze backbone up to layer name x.')
    # Re-Initialize model after each experience
    parser.add_argument('--reinit_model', action='store_true', default=False, help='Re-initialize model after each experience.')
    parser.add_argument('--reinit_after_exp', type=int, default=0, help='Re-initialize model after experience n.')
    parser.add_argument('--reinit_up_to_exp', type=int, default=None, help='Re-initialize model only up to experience n.')
    parser.add_argument('--reinit_layers_after', type=str, default=None, help='Reinit backbone after layer name x.')
    parser.add_argument('--reinit_freeze', action='store_true', default=False, help='Freeze backbone after reinit. This is complementary to freeze flag.')
   
    # Store model every experience
    parser.add_argument('--store_models', action='store_true', default=False, help='Store model after each experience.')

    # ER_AGEM_Custom
    parser.add_argument('--sample_size', default=0, type=int, 
                        help='Sample size to fine reference gradients for AGEM. Defaults to 0 which is deactivation.')
    parser.add_argument('--lmbda', type=float, default=0.5, help='Weighting of losses')
    parser.add_argument('--do_decay_lmbda', action='store_true', default=False, help='Decay lambda.')
    parser.add_argument('--lmbda_warmup', type=int, default=0, help='Number of warmup steps for lmbda in each experience.')
    return parser