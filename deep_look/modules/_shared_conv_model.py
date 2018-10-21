import torch


class SharedConvModel(torch.nn.Module):
    def __init__(self, input_feature_size, feature_size, num_layers=2):
        block = []
        for i in range(num_layers):
            fin = input_feature_size if i == 0 else feature_size

            block.append(torch.nn.ReLU(inplace=True))
            block.append(torch.nn.Conv2d(
                fin,
                feature_size,
                kernel_size=3,
                stride=1,
                padding=1
            ))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
