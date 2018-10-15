import math
import torch
form ..utils import anchors as utils_anchors


if torch.__version__ in ['0.4.1', '0.4.1.post2']:
    interpolate = torch.nn.functional.interpolate
else:
    interpolate = torch.nn.functional.upsample


conv_1x1_settings = {
    'kernel_size': 1,
    'stride': 1,
    'padding': 0
}

def _init_zero(t):
    torch.nn.init.constant_(t, 0.0)

def _init_uniform(t):
    torch.nn.init.normal_(t, 0.0, 0.01)


class FeaturePyramidNetwork(torch.nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/abs/1612.03144
    """
    def __init__(
        self,
        backbone_channel_sizes,
        min_feature_level=1,
        max_feature_level=7,
        feature_size=256,
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.min_feature_level = min_feature_level
        self.max_feature_level = max_feature_level

        C1_size, C2_size, C3_size, C4_size, C5_size = backbone_channel_sizes[-5:]

        self.conv_C5_reduce = torch.nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.conv_P5 = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        if self.min_feature_level <= 4:
            self.conv_C4_reduce = torch.nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
            self.conv_P4 = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        if self.min_feature_level <= 3:
            self.conv_C3_reduce = torch.nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
            self.conv_P3 = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        if self.min_feature_level <= 2:
            self.conv_C2_reduce = torch.nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
            self.conv_P2 = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        if self.min_feature_level <= 1:
            self.conv_C1_reduce = torch.nn.Conv2d(C1_size, feature_size, kernel_size=1, stride=1, padding=0)
            self.conv_P1 = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        if self.max_feature_level >= 6:
            self.conv_P6 = torch.nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        if self.max_feature_level >= 7:
            self.relu = torch.nn.ReLU(inplace=False)
            self.conv_P7 = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, C1, C2, C3, C4, C5):
        C5_reduced = self.conv_C5_reduce(C5)
        P5 = self.conv_P5(C5_reduced)
        features = (P5,)

        if self.min_feature_level <= 4:
            C5_upsampled = interpolate(C5_reduced, size=C4.shape[-2:], mode='bilinear', align_corners=False)
            C4_reduced = self.conv_C4_reduce(C4)
            P4 = self.conv_P4(C5_upsampled + C4_reduced)
            features = (P4,) + features

        if self.min_feature_level <= 3:
            C4_upsampled = interpolate(C4_reduced, size=C3.shape[-2:], mode='bilinear', align_corners=False)
            C3_reduced = self.conv_C3_reduce(C3)
            P3 = self.conv_P3(C4_upsampled + C3_reduced)
            features = (P3,) + features

        if self.min_feature_level <= 2:
            C3_upsampled = interpolate(C3_reduced, size=C2.shape[-2:], mode='bilinear', align_corners=False)
            C2_reduced = self.conv_C2_reduce(C2)
            P2 = self.conv_P2(C3_upsampled + C2_reduced)
            features = (P2,) + features

        if self.min_feature_level <= 1:
            C2_upsampled = interpolate(C2_reduced, size=C1.shape[-2:], mode='bilinear', align_corners=False)
            C1_reduced = self.conv_C1_reduce(C1)
            P1 = self.conv_P1(C2_upsampled + C1_reduced)
            features = (P1,) + features

        if self.max_feature_level >= 6:
            P6 = self.conv_P6(C5)
            features = features + (P6,)

        if self.max_feature_level >= 7:
            P7 = self.relu(P6)
            P7 = self.conv_P7(P7)
            features = features + (P7,)

        return features


class ComputeAnchors(torch.nn.Moudle):
    """
    Module that stores anchor generation instructions for ease of computation
    """
    def __init__(
        self,
        min_feature_level,
        max_feature_level,
        size_mult,
        stride_mult,
        ratios=[0.5, 1., 2.],
        scales=[2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]
    ):
        super(ComputeAnchors, self).__init__()

        self.stride = {
            i: 2 ** i * stride_mult
            for i in range(min_feature_level, max_feature_level + 1)
        }

        self.anchors = torch.nn.ParameterDict({
            i: utils_anchors.generate_anchors_at_window(2 ** i * size_mult)
            for i in range(min_feature_level, max_feature_level + 1)
        })
        for anchor in self.anchors.values():
            anchor.requires_grad = False

    def forward(self, feature_shape, feature_level):
        return utils_anchors.shift_anchors(
            feature_shape,
            self.stride[feature_level],
            self.anchors[feature_level]
        )


class DefaultRegressionModel(torch.nn.Module):
    def __init__(
        self,
        num_anchors,
        pyramid_feature_size=256,
        feature_size=256,
        num_layers=2
    ):
        super(DefaultRegressionModel, self).__init__()
        block = []

        # Create input layer
        block.append(torch.nn.Conv2d(
            pyramid_feature_size,
            feature_size,
            bias=True,
            **conv_1x1_settings
        ))
        block.append(torch.nn.ReLU(inplace=True))

        # Create intermediate layers
        for _ in range(num_layers - 1):
            block.append(torch.nn.Conv2d(
                feature_size,
                feature_size,
                bias=True,
                **conv_1x1_settings
            ))
            block.append(torch.nn.ReLU(inplace=True))

        # Create output layer
        block.append(torch.nn.Conv2d(
            feature_size,
            num_anchors * 4,
            bias=False,
            **conv_1x1_settings
        ))

        # Initialize regression output to be small
        _init_uniform(block[-1].weight)

        # Transform block into Sequential
        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)


class DefaultClassificationModel(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_anchors,
        pyramid_feature_size=256,
        feature_size=256,
        num_layers=2,
        prior_probability=0.01
    ):
        super(DefaultClassificationModel, self).__init__()
        self.num_classes = num_classes
        block = []

        # Create input layer
        block.append(torch.nn.Conv2d(
            pyramid_feature_size,
            feature_size,
            bias=True,
            **conv_1x1_settings
        ))
        block.append(torch.nn.ReLU(inplace=True))

        # Create intermediate layers
        for _ in range(num_layers - 1):
            block.append(torch.nn.Conv2d(
                feature_size,
                feature_size,
                bias=True,
                **conv_1x1_settings
            ))
            block.append(torch.nn.ReLU(inplace=True))

        # Create output layer
        block.append(torch.nn.Conv2d(
            feature_size,
            num_anchors * num_classes,
            bias=False,
            **conv_1x1_settings
        ))
        block.append(torch.nn.Sigmoid())

        # Initialize classification output to 0.01
        # kernel ~ 0.0
        # bias   ~ -log((1 - 0.01) / 0.01)  So that output is 0.01 after sigmoid
        kernel = block[-2].weight
        bias   = block[-2].bias
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - 0.01) / 0.01))

        # Transform block into Sequential
        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.num_classes)
