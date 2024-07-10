import torch.nn as nn
import math


def pool_out(in_size, kernel, dilation=1, padding=0, stride=None):
    stride = kernel if stride is None else stride

    out_size = (in_size + 2 * padding - dilation * (kernel-1) - 1) / stride + 1

    return int(math.floor(out_size))


class Classifier(nn.Module):
    def __init__(self, img_size, nf, num_classes):
        super(Classifier, self).__init__()
        nc, nh, nw = img_size
        self.blocks = nn.ModuleList()

        n_in = nc
        for i in range(len(nf)):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(n_in, nf[i], 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))

            nh = pool_out(nh, 2)
            nw = pool_out(nw, 2)
            n_in = nf[i]

        self.blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(nh * nw * n_in, 1 if num_classes == 2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
        ))

    def forward(self, x, output_feature_maps=False):
        intermediate_outputs = []

        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
