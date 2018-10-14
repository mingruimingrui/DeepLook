"""
Dynamic backbone model loader
More backbone models other than resnet might be added in the future but currently
there are no plans to support such a feature
"""
import torch
import torchvision


class ResNetBackbone(torch.nn.Module):
    def __init__(self, backbone_name, freeze_backbone=False, freeze_batchnorm=True):
        super(ResNetBackbone, self).__init__()

        # Load a pretrained resnet model
        resnet_model = getattr(torchvision.models, backbone_name)(pretrained=True)

        # Copy layers with weights
        self.conv1   = resnet_model.conv1
        self.bn1     = resnet_model.bn1
        self.relu    = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1  = resnet_model.layer1
        self.layer2  = resnet_model.layer2
        self.layer3  = resnet_model.layer3
        self.layer4  = resnet_model.layer4

        # Delete unused layers
        del resnet_model.avgpool
        del resnet_model.fc
        del resnet_model

        # Freeze batch norm
        if freeze_batchnorm:
            for layer in self.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.track_running_stats = False
                    for param in layer.parameters():
                        param.requires_grad = False

        # Freeze backbone if flagged
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Get layers which outputs C3, C4, C5
        C1 = self.conv1(x)
        C1 = self.bn1(C1)
        C1 = self.relu(C1)

        C2 = self.maxpool(C1)
        C2 = self.layer1(C2)

        C3 = self.layer2(C2)

        C4 = self.layer3(C3)

        C5 = self.layer4(C4)

        return C1, C2, C3, C4, C5


def get_backbone_channel_sizes(backbone_model):
    dummy_input = torch.zeros(1, 3, 224, 224)
    dummy_outputs = backbone_model(dummy_input)
    return [o.shape[1] for o in dummy_outputs]


def Backbone(backbone_name, freeze_backbone=False, freeze_batchnorm=True):
    if 'resnet' in backbone_name:
        return ResNetBackbone(backbone_name, freeze_backbone=freeze_backbone, freeze_batchnorm=freeze_batchnorm)
    else:
        raise NotImplementedError('{} has not been implemented yet'.format(backbone_name))
