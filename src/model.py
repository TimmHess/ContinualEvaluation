#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from __future__ import division
from turtle import forward

from typing import TYPE_CHECKING, Optional, List
from typing import NamedTuple, List, Optional, Tuple, Callable
from collections import OrderedDict

import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.functional import relu, avg_pool2d, max_pool2d

from torchvision.models import resnet18, ResNet18_Weights

from torchinfo import summary

from avalanche.models.dynamic_modules import MultiHeadClassifier

def get_feat_size(block, spatial_size, in_channels=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.
    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    Source: https://github.com/TimmHess/OCDVAEContinualLearning/blob/master/lib/Models/architectures.py
    """

    x = torch.randn(2, in_channels, spatial_size, spatial_size)
    out = block(x)
    if len(out.size()) == 2: # NOTE: catches the case where the block is a linear layer
        num_feat = out.size(1)
        spatial_dim_x = 1
        spatial_dim_y = 1
    else:
        num_feat = out.size(1)
        spatial_dim_x = out.size(2)
        spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y

def initialize_weights(m) -> None:
    """
    Initilaize weights of model m.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    return 

class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor

def get_layers_and_params(model, prefix=''):
    """
    Adapted from AvalancheCL lib
    """
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

def freeze_up_to(model: Module,
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
    for param_def in get_layers_and_params(model, prefix=module_prefix):
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


'''
Layer Definitions
'''
class L2NormalizeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return torch.nn.functional.normalize(x, p=2, dim=1)


class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class FeatAvgPoolLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """ This format to be compatible with OpenSelfSup and the classifiers expecting a list."""
        # Pool
        assert x.dim() == 4, \
            "Tensor must has 4 dims, got: {}".format(x.dim())
        x = self.avg_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out



'''
Backbones
'''
class SimpleCNNFeat(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNNFeat, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(p=0.25)
        )

        self.feature_size = self.calc_feature_size(input_size)
    
    def calc_feature_size(self, input_size):
        self.feature_size = self.features(torch.zeros(1, *input_size)).view(1, -1).size(1)
        return self.feature_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x



class VGG11ConvBlocks(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # Flatten feature maps
            nn.Flatten()
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

class VGG11DenseBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.features = nn.Sequential(
            # # Dense
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            # Dense 
            nn.Linear(1024, 128)
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

class VGG11Feat(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.conv_blocks = VGG11ConvBlocks(input_size)
        self.conv_block_out_size = self.conv_blocks(torch.zeros(1, *input_size)).view(1, -1).size(1)

        self.dense_block = VGG11DenseBlock(self.conv_block_out_size)

        self.features = nn.Sequential(
           self.conv_blocks,
           self.dense_block
        )

        self.feature_size = self.calc_feature_size(input_size)
        print("vgg11 feature_size:", self.feature_size)
        return

    def calc_feature_size(self, input_size):
        self.feature_size = self.features(torch.zeros(1, *input_size)).size(1)
        return self.feature_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


class MLPfeat(nn.Module):
    def_hidden_size = 400

    def __init__(self, nonlinear_embedding: bool, input_size=28 * 28,
                 hidden_sizes: tuple = None, nb_layers=2):
        """
        :param nonlinear_embedding: Include non-linearity on last embedding layer.
        This is typically True for Linear classifiers on top. But is false for embedding based algorithms.
        :param input_size:
        :param hidden_size:
        :param nb_layers:
        """
        super().__init__()
        assert nb_layers >= 2
        if hidden_sizes is None:
            hidden_sizes = [self.def_hidden_size] * nb_layers
        else:
            assert len(hidden_sizes) == nb_layers
        self.feature_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes

        # Need at least one non-linear layer
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_sizes[0]),
                                 nn.ReLU(inplace=True)
                                 ))

        for layer_idx in range(1, nb_layers - 1):  # Not first, not last
            layers.add_module(
                f"fc{layer_idx}", nn.Sequential(
                    *(nn.Linear(hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]),
                      nn.ReLU(inplace=True)
                      )))

        # Final layer
        layers.add_module(
            f"fc{nb_layers}", nn.Sequential(
                *(nn.Linear(hidden_sizes[nb_layers - 2],
                            hidden_sizes[nb_layers - 1]),
                  )))

        # Optionally add final nonlinearity
        if nonlinear_embedding:
            layers.add_module(
                f"final_nonlinear", nn.Sequential(
                    *(nn.ReLU(inplace=True),)))

        self.features = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        # x = self.classifier(x)
        return x

'''
ResNet
'''
class ResNet(nn.Module):
    """ ResNet feature extractor, slimmed down according to GEM paper."""

    def __init__(self, block, num_blocks, nf, global_pooling, input_size):
        """

        :param block:
        :param num_blocks:
        :param nf: Number of feature maps in each conv layer.
        """
        super(ResNet, self).__init__()
        self.global_pooling = global_pooling

        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.feature_size = None

        assert len(input_size) >= 3
        input_size = input_size[-3:]  # Only care about last 3

        if nf==20:
            if input_size == (3, 32, 32):  # Cifar10
                self.feature_size = 160 if global_pooling else 2560
            elif input_size == (3, 84, 84):  # Mini-Imagenet
                self.feature_size = 640 if global_pooling else 19360
            elif input_size == (3, 96, 96):  # TinyDomainNet
                self.feature_size = 1440 if global_pooling else 23040
            else:
                raise ValueError(f"Input size not recognized: {input_size}")
        else:
            pass

        # self.linear = nn.Linear(self.feature_size, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.shape) == 4, "Assuming x.view(bsz, C, W, H)"
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.global_pooling:
            out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # Flatten
        # out = self.linear(out)
        return out


class ResNetPT(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNetPT, self).__init__()
        self.pretrained = pretrained
        if self.pretrained:
            print("\nUsing pretrained resnet18 model weights from pytorch!")
            self.features = torch.nn.Sequential(*(list((resnet18(weights=ResNet18_Weights.DEFAULT)).children())[:-1]))
        else:
            print("\nUsing resnet18 as by pytorch!")
            self.features = torch.nn.Sequential(*(list((resnet18()).children())[:-1]))
        self.feature_size = None
        return

    def forward(self, x):
        assert len(x.shape) == 4, "Assuming x.view(bsz, C, W, H)"
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten
        return out



def ResNet18feat(input_size, nf=20, global_pooling=False, use_torch_version=False, pretrained=False):
    model_backbone = None
    if use_torch_version:
        model_backbone = ResNetPT(pretrained=pretrained)
        print("\nWill use pretrained model")
    else:
        model_backbone = ResNet(BasicBlock, [2, 2, 2, 2], nf, global_pooling=global_pooling, input_size=input_size)
    enc_channels, enc_spatial_dim_x, enc_spatial_dim_y = get_feat_size(model_backbone, spatial_size=input_size[1], in_channels=input_size[0])
    model_backbone.feature_size = enc_channels * enc_spatial_dim_x * enc_spatial_dim_y
    print("\nModel Feature Size:", model_backbone.feature_size)
    return model_backbone


'''
Wide ResNet
'''
class WRNBasicBlock(nn.Module):
    """
    Convolutional or transposed convolutional block consisting of multiple 3x3 convolutions with short-cuts,
    ReLU activation functions and batch normalization.
    """
    def __init__(self, in_planes, out_planes, stride, batchnorm=1e-5, is_transposed=False):
        super(WRNBasicBlock, self).__init__()

        if is_transposed:
            self.layer1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                             output_padding=int(stride > 1), bias=False)
        else:
            self.layer1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.useShortcut = ((in_planes == out_planes) and (stride == 1))
        if not self.useShortcut:
            if is_transposed:
                self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                                   output_padding=int(1 and stride == 2), bias=False)
            else:
                self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.useShortcut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.layer1(out if self.useShortcut else x)))
        out = self.conv2(out)

        return torch.add(x if self.useShortcut else self.shortcut(x), out)


class WRNNetworkBlock(nn.Module):
    """
    Convolutional or transposed convolutional block
    """
    def __init__(self, nb_layers, in_planes, out_planes, block_type, batchnorm=1e-5, stride=1,
                 is_transposed=False):
        super(WRNNetworkBlock, self).__init__()

        if is_transposed:
            self.block = nn.Sequential(OrderedDict([
                ('convT_block' + str(layer + 1), block_type(layer == 0 and in_planes or out_planes, out_planes,
                                                             layer == 0 and stride or 1, batchnorm=batchnorm,
                                                             is_transposed=(layer == 0)))
                for layer in range(nb_layers)
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv_block' + str(layer + 1), block_type((layer == 0 and in_planes) or out_planes, out_planes,
                                                           (layer == 0 and stride) or 1, batchnorm=batchnorm))
                for layer in range(nb_layers)
            ]))

    def forward(self, x):
        x = self.block(x)
        return x


class WRN(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting and to our unified model.
    NOTE: Default values are taken from: https://github.com/MrtnMndt/OpenVAE_ContinualLearning/blob/master/lib/cmdparser.py
    """
    def __init__(self, input_size, use_bn=True, wrn_embedding_size=48, 
            widen_factor=10, depth=14, use_max_pool=False):
        super(WRN, self).__init__()

        self.input_size = input_size
        self.use_bn = 1e-5 if use_bn else 0.0
        self.use_max_pool = use_max_pool

        self.widen_factor = widen_factor
        self.depth = depth
        self.wrn_embedding_size = wrn_embedding_size

        self.nChannels = [self.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,
                          64 * self.widen_factor]

        assert ((self.depth - 2) % 6 == 0)
        self.num_block_layers = int((self.depth - 2) / 6)

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(self.input_size[0], self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
            ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                               WRNBasicBlock, batchnorm=self.use_bn)),
            ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                               WRNBasicBlock, batchnorm=self.use_bn, stride=2)),
            ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                               WRNBasicBlock, batchnorm=self.use_bn, stride=2)),
            ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.use_bn)),
            ('encoder_act1', nn.ReLU(inplace=True))
        ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = \
            get_feat_size(self.encoder, spatial_size=self.input_size[1], in_channels=self.input_size[0])

        self.feature_size = self.enc_channels * self.enc_spatial_dim_x * self.enc_spatial_dim_y

    def forward(self, x):
        x = self.encoder(x)
        
        if self.use_max_pool:
            x = max_pool2d(x, kernel_size=4)

        x = x.view(x.size(0), -1) # Flatten
        return x

def WRNfeat(input_size, wrn_embedding_size=48, widen_factor=10, depth=14, 
        use_bn=False, use_max_pool=False):
    return WRN(input_size=input_size, wrn_embedding_size=wrn_embedding_size, 
        widen_factor=widen_factor, depth=depth,
        use_bn=use_bn, use_max_pool=use_max_pool)

'''
Classifier
'''
class FeatClassifierModel(torch.nn.Module):
    def __init__(self, feature_extractor, classifier, with_adaptive_pool=False):
        super().__init__()
        self.with_adaptive_pool = with_adaptive_pool

        self.feature_extractor = feature_extractor
        self.classifier = classifier  # Linear or MultiTaskHead

        self.last_features = None

        if self.with_adaptive_pool:
            self.avg_pool = FeatAvgPoolLayer()

    def forward_feats(self, x):
        x = self.feature_extractor(x)
        if self.with_adaptive_pool:
            x = self.avg_pool(x)
        # store last computed features
        self.last_features = x
        return x

    def forward_classifier(self, x, task_labels=None):
        try:  # Multi-task head
            x = self.classifier(x, task_labels)
        except:  # Single head
            x = self.classifier(x)
        return x

    def forward(self, x, task_labels=None):
        x = self.forward_feats(x)
        x = self.forward_classifier(x, task_labels)
        return x


########################################################################################################################
def get_model(args, n_classes):
    """ Build model from feature extractor and classifier."""
    feat_extr = _get_feat_extr(args)  # Feature extractor
    classifier = _get_classifier(args, n_classes, feat_extr.feature_size)  # Classifier
    model = FeatClassifierModel(feat_extr, classifier)  # Combined model
    print("\nBackbone Summary:")
    summary(feat_extr, input_size=(1, args.input_size[0], args.input_size[1], args.input_size[1]))
    if args.show_backbone_param_names:
        print("Modules:")
        for module in feat_extr.named_modules():
            print(module)
        print("\nStopping execution here! Remove the 'show_backbone_param_names' flag to continue!")
        import sys;sys.exit()
    return model


def _get_feat_extr(args):
    """ Get embedding network. """
    nonlin_embedding = args.classifier in ['linear']  # Layer before linear should have nonlinearities
    input_size = math.prod(args.input_size)

    if args.backbone == "mlp":  # MNIST mlp
        feat_extr = MLPfeat(hidden_sizes=(400, args.featsize), nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, input_size=input_size)
    elif args.backbone == 'simple_cnn':
        feat_extr = SimpleCNNFeat(input_size=args.input_size)
    elif args.backbone == 'vgg11':
        feat_extr = VGG11Feat(input_size=args.input_size)
    elif args.backbone == "resnet18":
        feat_extr = ResNet18feat(nf=20, global_pooling=args.use_GAP, input_size=args.input_size)
    elif args.backbone == "resnet18_big":
        feat_extr = ResNet18feat(nf=64, global_pooling=args.use_GAP, input_size=args.input_size)
    elif args.backbone == "resnet18_big_t": # torch version
        feat_extr = ResNet18feat(nf=64, global_pooling=args.use_GAP, input_size=args.input_size, use_torch_version=True, pretrained=False)
    elif args.backbone == "resnet18_big_pt": # torch version - pretrained
        feat_extr = ResNet18feat(nf=64, global_pooling=args.use_GAP, input_size=args.input_size, use_torch_version=True, pretrained=True)
    elif args.backbone == 'wrn':
        feat_extr = WRNfeat(args.input_size, wrn_embedding_size=args.wrn_embedding_size, widen_factor=args.wrn_widen_factor, 
            depth=args.wrn_depth, use_bn=False, use_max_pool=args.use_maxpool)
    else:
        raise ValueError()
    #assert hasattr(feat_extr, 'feature_size'), "Feature extractor requires attribute 'feature_size'"
    return feat_extr


def _get_classifier(args, n_classes, feat_size):
    """ Get classifier head. For embedding networks this is normalization or identity layer."""
    # No prototypes, final linear layer for classification
    if args.classifier == 'linear':  # Lin layer
        if args.task_incr:
            classifier = MultiHeadClassifier(in_features=feat_size,
                                             initial_out_features=args.initial_out_features,
                                             use_bias=args.lin_bias)
        else:
            classifier = torch.nn.Linear(in_features=feat_size, out_features=n_classes, bias=args.lin_bias)
    # Prototypes held in strategy
    elif args.classifier == 'norm_embed':  # Get feature normalization
        classifier = L2NormalizeLayer()
    elif args.classifier == 'identity':  # Just extract embedding output
        classifier = torch.nn.Flatten()
    else:
        raise NotImplementedError()
    return classifier



########################################################################################################################
if __name__ == "__main__":
    import torch

    bs = 5

    x_cifar = torch.rand((bs, 3, 32, 32))
    x_miniimg = torch.rand((bs, 3, 84, 84))
    x_tinydomainnet = torch.rand((bs, 3, 96, 96))

    # # NO GAP
    # model = ResNet18feat(nf=20, global_pooling=False, input_size=None)

    # cifar_shape = model.forward(x_cifar).shape  # 2560
    # mini_shape = model.forward(x_miniimg).shape  # 19360
    # tiny_shape = model.forward(x_tinydomainnet).shape  # 23040

    # # WITH GAP
    # model = ResNet18feat(nf=20, global_pooling=True, input_size=None)

    # cifar_shape_gap = model.forward(x_cifar).shape  # 160
    # mini_shape_gap = model.forward(x_miniimg).shape  # 640
    # tiny_shape_gap = model.forward(x_tinydomainnet).shape  # 1440

    # model = WRNfeat(input_size=(3,32,32), use_max_pool=True)
    # shape = model.forward(x_cifar).shape
    # print(shape)

    # model = WRNfeat(input_size=(3,84,84), use_max_pool=True)
    # shape = model.forward(x_miniimg).shape
    # print(shape)

    # model = WRNfeat(input_size=(3,96,96), use_max_pool=True)
    # shape = model.forward(x_tinydomainnet).shape
    # print(shape)
