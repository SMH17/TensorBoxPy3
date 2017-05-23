# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

"""Brings all resnet models under one namespace."""

# pylint: disable=unused-import
from utils.slim_nets.resnet_v1 import resnet_v1
from utils.slim_nets.resnet_v1 import resnet_v1_50
from utils.slim_nets.resnet_v1 import resnet_v1_101
from utils.slim_nets.resnet_v1 import resnet_v1_152
from utils.slim_nets.resnet_v1 import resnet_v1_200
from utils.slim_nets.resnet_v2 import resnet_v2
from utils.slim_nets.resnet_v2 import resnet_v2_50
from utils.slim_nets.resnet_v2 import resnet_v2_101
from utils.slim_nets.resnet_v2 import resnet_v2_152
from utils.slim_nets.resnet_v2 import resnet_v2_200
from utils.slim_nets.resnet_utils import resnet_arg_scope
# pylint: enable=unused-import
