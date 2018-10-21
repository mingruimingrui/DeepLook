import torch


class ObjectFinderNetwork(torch.nn.Module):
    """
    Object finder network which identifies if anchor position or part therefore contains any objects
    """
    def __init__(
        self,
        pyramid_feature_size=256,
        feature_size=256,
        num_layers=2
    ):
        super(ObjectFinderNetwork, self).__init__()

        block = []
        for i in range(num_layers + 1):
            feature_in = pyramid_feature_size if i == 0 else feature_size
            feature_out = 1 if i == num_layers else feature_size
            activation = torch.nn.Sigmoid() if i == num_layers else torch.nn.ReLU(inplace=True)

            block.append(activation)
            block.append(torch.nn.Conv2d(
                feature_in,
                feature_out,
                kernel_size=1,
                stride=1,
                padding=0
            ))
            block.append(activation)

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
