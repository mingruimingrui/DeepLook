""" DeepLook implementation in pytorch """

import torch

# Default configs
from ._deep_look_configs import make_configs

# Other modules
from ._backbone import (
    Backbone,
    get_backbone_channel_sizes
)

from ._feature_pyramid_network import FeaturePyramidNetwork
from ._object_finder_network import ObjectFinderNetwork


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

        self.shared_conv_model = SharedConvModel(
            input_feature_size=self.configs['pyramid_feature_size'],
            feature_size=self.configs['shared_conv_feature_size'],
            num_layers=self.configs['shared_conv_num_layers']
        )

        if self.configs['shared_conv_num_layers'] > 0:
            shared_conv_output_size = self.configs['shared_conv_feature_size']
        else:
            shared_conv_output_size = self.configs['pyramid_feature_size']

        self.ofn = ObjectFinderNetwork(
            input_feature_size=shared_conv_output_size,
            feature_size=self.configs['finder_feature_size'],
            num_layers=self.configs['finder_num_layers']
        )

        self.ofn_loss_fn

        # self.classification_model = ClassificationModel()
        #
        # self.regression_model = RegressionModel()

    # def _recursive_forward(self, features):
    #     if self.training:
    #         for level in range(self.configs['max_feature_level'], self3.configs['min_feature_level'] - 1, -1):
    #             self.ofn(features[level])

    def forward(self, image_batch, annotations_batch=None):
        # if self.training:
        #     assert annotations is not None

        image_batch_shape = image_batch.shape
        batch_size = image_batch_shape[0]
        image_hw = image_batch_shape[-2:]

        # Similar to the retinanet, generate multi level features with FPN
        backbone_output = self.backbone(image_batch)
        features = self.fpn(*backbone_output)

        if self.configs['shared_conv_num_layers'] > 0:
            for level, feature in features.items():
                features[level] = self.shared_conv_model(feature)

        found_object_pos = {}
        for level, feature in features.items():
            found_object_pos[level] = self.ofn[feature]

        if self.training:
            self.loss_fn =

        return None
