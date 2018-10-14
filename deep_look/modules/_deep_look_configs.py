from __future__ import division
import warnings
from copy import deepcopy
from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

#### Model configs
_c.name        = 'DeepLook'  # This variable does nothing, only for model loading
_c.num_classes = None   # int, 1 or greater

_c.backbone        = 'resnet50'
_c.freeze_backbone = False

_c.min_feature_level = 1  # int, between 1-7 inclusive, equal or smaller than max_feature_level
_c.max_feature_level = 7  # int, between 1-7 inclusive, equal or larger than min_feature_level
_c.pyramid_feature_size   = 256   # int, 1 or greater

_c.anchor_size_mult   = 4.0  # multiplier for anchor size, at feature level n, anchor size will be (2 ** n) * mult
_c.anchor_stride_mult = 1.0  # multiplier for anchor stride, at feature level n, anchor size will be (2 ** n) * mult

_c.anchor_ratios  = [0.5, 1., 2.]
_c.anchor_scales  = [2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]

_c.regression_num_layers   = 2  # int, 1 or greater
_c.regression_feature_size = 256  # Regression model internal channel size

_c.classification_num_layers   = 2  # int, 1 or greater
_c.classification_feature_size = 256  # Classification model internal channel size

_c.apply_nms       = True  # If True, nms will be applied on detections, won't if False
_c.nms_use_cpu     = True  # NMS can be memory inefficient
_c.nms_type        = 'hard'  # currently only 'hard' accepted
_c.nms_threshold   = 0.5
_c.score_threshold = 0.05
_c.max_detections  = 300

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def validate_configs(configs):
    """ Verify that configs are valid and performs feature generation """
    assert isinstance(configs.num_classes, int), 'num_classes must be specified'
    assert 'resnet' in configs.backbone, 'only resnet backbones supported'

    assert configs.min_feature_level >= 1, 'min_feature_level must be 1 or greater'
    assert configs.min_feature_level <= configs.max_feature_level, 'max_feature_level must be atleast min_feature_level'
    assert configs.max_feature_level <= 7, 'max_feature_level must be 7 or less'

    # TODO: Determine the minimum bounds of each variable

    configs.num_anchors = len(configs.anchor_ratios) * len(configs.anchor_scales)

def make_configs(**kwargs):
    configs = deepcopy(_c)
    configs.immutable(False)

    # Update default configs with user provided ones
    for arg, value in kwargs.items():
        if arg == 'name':
            warnings.warn('DeepLook model name cannot be changed, skipping')
        elif arg not in configs:
            warnings.warn('{} is not a valid arg for DeeLook, skipping'.format(arg))
        else:
            configs[arg] = value

    validate_configs(configs)
    configs.immutable(True)

    return configs
