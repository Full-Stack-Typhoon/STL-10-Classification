#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sslime.models.trunks.alexnet_rotnet import AlexNet_RotNet
from sslime.models.trunks.resnet50 import ResNet50
from sslime.models.trunks.vgg_a import VGG_A

TRUNKS = {"alexnet": AlexNet_RotNet, "resnet50": ResNet50, "vgg_a":VGG_A}
