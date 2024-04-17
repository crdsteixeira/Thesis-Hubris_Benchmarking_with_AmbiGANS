from .pretrained import ClassifierVIT, ClassifierResnet, ClassifierMLP
from .simple_cnn import Classifier
import torch.nn as nn
import ast
import torch

class Ensemble(nn.Module):
    def __init__(self, img_size, num_classes, nf, ensemble_type, device):
        super(Ensemble, self).__init__()
        self.num_channels = img_size[0]
        self.device = device

        # List of ensemble models, either pretrained ones or CNN's.
        if ensemble_type == "pretrained":
            self.models = nn.ModuleList(
                [
                    ClassifierResnet(img_size, num_classes, nf, self.device),
                    ClassifierMLP(img_size, num_classes, nf, self.device),
                ],
            )
        elif ensemble_type == "cnn":
            # Generate models' parameters
            self.cnn_list = nf
            self.cnn_count = len(self.cnn_list)
            # Generate models
            self.models = nn.ModuleList(
                [
                    Classifier(
                        img_size,
                        nf=cnn,
                        num_classes=num_classes
                    ) for cnn in self.cnn_list
                ]
            )

        # Multi-probability combinator.
        self.predictor = nn.Sequential(
            nn.Linear(len(self.models), 1),
            nn.Sigmoid(),
        )

    def forward(self, x, output_feature_maps=False):
        output = torch.Tensor().to(self.device)
        feat_maps = []
        for m in self.models:
            out = m(x.clone(), output_feature_maps=False).unsqueeze(-1)
            output = torch.hstack((output, out))

        feat_maps.append(output)
        output = self.predictor(output).squeeze(1)
        feat_maps.append(output)

        if output_feature_maps:
            return feat_maps
        else:
            return output