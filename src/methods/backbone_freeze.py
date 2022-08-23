import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from typing import TYPE_CHECKING, Optional, List
from typing import NamedTuple, List, Optional, Tuple, Callable

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import freeze_everything, freeze_up_to, get_layers_and_params
from src.utils import get_grad_normL2


class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor


class FreezeBackbonePlugin(StrategyPlugin):
    def __init__(self, exp_to_freeze_on=0, freeze_up_to_layer_name=None):
        super().__init__()

        self.exp_to_freeze_on = exp_to_freeze_on
        self.freeze_up_to_layer_name = freeze_up_to_layer_name
        return

    def get_layers_and_params(self, model, prefix=''):
        result: List[LayerAndParameter] = []
        layer_name: str
        layer: Module
        for layer_name, layer in model.named_modules():
            if layer == model:
                continue
            if isinstance(layer, nn.Sequential): # NOTE: cannot unclude Sequentials because this is basically a repetition of parameter listings
                continue
            layer_complete_name = prefix + layer_name + "."

            layers_and_params = get_layers_and_params(layer, prefix=layer_complete_name) #NOTE: this calls to avalanche function! (not self)
            result += layers_and_params
        return result

    
    def freeze_up_to(self, model: Module,
        freeze_until_layer: str = None,
        set_eval_mode: bool = True,
        set_requires_grad_false: bool = True,
        layer_filter: Callable[[LayerAndParameter], bool] = None,
        module_prefix: str = "",):
        """
        A simple utility that can be used to freeze a model.
        :param model: The model.
        :param freeze_until_layer: If not None, the freezing algorithm will continue
            (proceeding from the input towards the output) until the specified layer
            is encountered. The given layer is excluded from the freezing procedure.
        :param set_eval_mode: If True, the frozen layers will be set in eval mode.
            Defaults to True.
        :param set_requires_grad_false: If True, the autograd engine will be
            disabled for frozen parameters. Defaults to True.
        :param layer_filter: A function that, given a :class:`LayerParameter`,
            returns `True` if the parameter must be frozen. If all parameters of
            a layer are frozen, then the layer will be set in eval mode (according
            to the `set_eval_mode` parameter. Defaults to None, which means that all
            parameters will be frozen.
        :param module_prefix: The model prefix. Do not use if non strictly
            necessary.
        :return:
        """

        frozen_layers = set()
        frozen_parameters = set()

        to_freeze_layers = dict()
        for param_def in self.get_layers_and_params(model, prefix=module_prefix):
            if (
                freeze_until_layer is not None
                and freeze_until_layer == param_def.layer_name
            ):
                break

            freeze_param = layer_filter is None or layer_filter(param_def)
            if freeze_param:
                if set_requires_grad_false:
                    param_def.parameter.requires_grad = False
                    frozen_parameters.add(param_def.parameter_name)

                if param_def.layer_name not in to_freeze_layers:
                    to_freeze_layers[param_def.layer_name] = (True, param_def.layer)
            else:
                # Don't freeze this parameter -> do not set eval on the layer
                to_freeze_layers[param_def.layer_name] = (False, None)

        if set_eval_mode:
            for layer_name, layer_result in to_freeze_layers.items():
                if layer_result[0]:
                    layer_result[1].eval()
                    frozen_layers.add(layer_name)

        return frozen_layers, frozen_parameters

    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter > (self.exp_to_freeze_on -1): # NOTE: -1 is required to be able to freeze on the 0th experience
            print("\n\nFreezing backbone...\n\n")
            if self.freeze_up_to_layer_name is None:
                print("Freezing entire model...")
                freeze_everything(strategy.model.feature_extractor)
            else: 
                print("Freezing model up to layer {}...".format(self.freeze_up_to_layer_name))
                frozen_layers, _ = self.freeze_up_to(strategy.model.feature_extractor, self.freeze_up_to_layer_name)
                for layer_name in frozen_layers:
                    print("Froze layer: {}".format(layer_name))
        return