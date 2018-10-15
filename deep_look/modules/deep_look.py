""" DeepLook implementation in pytorch """

import torch

# Default configs
from ._deep_look_configs import make_configs

# Other modules
from ._backbone import (
    Backbone,
    get_backbone_channel_sizes
)

from ._deep_look import (
    FeaturePyramidNetwork
)


class DeepLook(torch.nn.Module):
    """
    DeepLook aims to improve upon the retinanet, it mainly targets 2 main areas
    of shortcoming
    - large memory requirements due to numerous anchors
    - inability to limit depth of search
        (for cases where only large objects are present)
    """
    def __init__(self, num_classes, **kwargs):
        """
        Constructs a DeepLook model with initialized weights
        """
        super(DeepLook, self).__init__()

        # Make config
        kwargs['num_classes'] = num_classes
        self.configs = make_configs(**kwargs)

        # Build submodules
        self.build_modules()

    def build_modules(self):
        """ Build all submodels for DeepLook """
        self.backbone = Backbone(
            self.configs['backbone'],
            freeze_backbone=self.configs['freeze_backbone'],
            freeze_batchnorm=True
        )

        backbone_channel_sizes = get_backbone_channel_sizes(self.backbone)

        self.fpn = FeaturePyramidNetwork(
            backbone_channel_sizes=backbone_channel_sizes,
            min_feature_level=self.configs['min_feature_level'],
            max_feature_level=self.configs['max_feature_level'],
            feature_size=self.configs['pyramid_feature_size']
        )

        self.compute_anchors = ComputeAnchors(
            min_feature_level=self.configs['min_feature_level'],
            max_feature_level=self.configs['max_feature_level'],
            size_mult=self.configs['anchor_size_mult'],
            stride_mult=self.configs['anchor_stride_mult'],
            ratios=self.configs['anchor_ratios'],
            scales=self.configs['anchor_scales']
        )

        self.regression_submodule = DefaultRegressionModel(
            num_anchors=self.confgs['num_anchors'],
            pyramid_feature_size=self.configs['pyramid_feature_size'],
            feature_size=self.configs['regression_feature_size'],
            num_layers=self.configs['regression_num_layers']
        )

        self.classification_submodule = DefaultClassificationModel(
            num_classes=self.configs['num_classes'],
            num_anchors=self.confgs['num_anchors'],
            pyramid_feature_size=self.configs['pyramid_feature_size'],
            feature_size=self.configs['classification_feature_size'],
            num_layers=self.configs['classification_feature_size_num_layers'],
            prior_probability=0.01
        )

    def forward(self, image_batch, annotations_batch=None):
        # if self.training:
        #     assert annotations is not None

        image_batch_shape = image_batch.shape
        batch_size = image_batch_shape[0]
        image_hw = image_batch_shape[-2:]

        backbone_output = self.backbone(image_batch)
        features = self.fpn.forward(*backbone_output)

        return None
