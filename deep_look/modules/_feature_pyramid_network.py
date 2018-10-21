import torch


if torch.__version__ in ['0.4.1', '0.4.1.post2']:
    interpolate = torch.nn.functional.interpolate
else:
    interpolate = torch.nn.functional.upsample


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
        assert min_feature_level >= 2
        assert max_feature_level <= 7
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
        features = { 5: P5 }

        if self.min_feature_level <= 4:
            C5_upsampled = interpolate(C5_reduced, size=C4.shape[-2:], mode='bilinear', align_corners=False)
            C4_reduced = self.conv_C4_reduce(C4)
            P4 = self.conv_P4(C5_upsampled + C4_reduced)
            features[4] = P4

        if self.min_feature_level <= 3:
            C4_upsampled = interpolate(C4_reduced, size=C3.shape[-2:], mode='bilinear', align_corners=False)
            C3_reduced = self.conv_C3_reduce(C3)
            P3 = self.conv_P3(C4_upsampled + C3_reduced)
            features[3] = P3

        if self.min_feature_level <= 2:
            C3_upsampled = interpolate(C3_reduced, size=C2.shape[-2:], mode='bilinear', align_corners=False)
            C2_reduced = self.conv_C2_reduce(C2)
            P2 = self.conv_P2(C3_upsampled + C2_reduced)
            features[2] = P2

        if self.min_feature_level <= 1:
            C2_upsampled = interpolate(C2_reduced, size=C1.shape[-2:], mode='bilinear', align_corners=False)
            C1_reduced = self.conv_C1_reduce(C1)
            P1 = self.conv_P1(C2_upsampled + C1_reduced)
            features[1] = P1

        if self.max_feature_level >= 6:
            P6 = self.conv_P6(C5)
            features[6] = P6

        if self.max_feature_level >= 7:
            P7 = self.relu(P6)
            P7 = self.conv_P7(P7)
            features[7] = P7

        return features
