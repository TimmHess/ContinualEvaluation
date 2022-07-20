#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from __future__ import division
from turtle import forward

from typing import List

import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

from avalanche.models.dynamic_modules import MultiHeadClassifier


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

        assert len(input_size) >= 3
        input_size = input_size[-3:]  # Only care about last 3

        if input_size == (3, 32, 32):  # Cifar10
            self.feature_size = 160 if global_pooling else 2560
        elif input_size == (3, 84, 84):  # Mini-Imagenet
            self.feature_size = 640 if global_pooling else 19360
        elif input_size == (3, 96, 96):  # TinyDomainNet
            self.feature_size = 1440 if global_pooling else 23040
        else:
            raise ValueError(f"Input size not recognized: {input_size}")

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


def ResNet18feat(input_size, nf=20, global_pooling=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], nf, global_pooling=global_pooling, input_size=input_size)



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
        # store last computed features
        self.last_features = x
        if self.with_adaptive_pool:
            x = self.avg_pool(x)
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

    # NO GAP
    model = ResNet18feat(nf=20, global_pooling=False, input_size=None)

    cifar_shape = model.forward(x_cifar).shape  # 2560
    mini_shape = model.forward(x_miniimg).shape  # 19360
    tiny_shape = model.forward(x_tinydomainnet).shape  # 23040

    # WITH GAP
    model = ResNet18feat(nf=20, global_pooling=True, input_size=None)

    cifar_shape_gap = model.forward(x_cifar).shape  # 160
    mini_shape_gap = model.forward(x_miniimg).shape  # 640
    tiny_shape_gap = model.forward(x_tinydomainnet).shape  # 1440

