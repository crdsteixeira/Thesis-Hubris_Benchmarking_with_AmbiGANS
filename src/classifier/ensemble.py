from .pretrained import ClassifierVIT, ClassifierResnet, ClassifierMLP
from .simple_cnn import Classifier
import torch.nn as nn
import ast
import torch

class Ensemble(nn.Module):
    def __init__(self, img_size, num_classes, nf, ensemble_type, output_method, device):
        super(Ensemble, self).__init__()
        self.num_channels = img_size[0]
        self.ensemble_type = ensemble_type
        self.output_method = output_method
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

        # Output method for ensemble
        if output_method == "super-learner":
            # Multi-probability combinator.
            self.predictor = nn.Sequential(
                nn.Linear(len(self.models), len(self.models)),
                nn.ReLU(),
                nn.Linear(len(self.models), len(self.models)),
                nn.ReLU(),
                nn.Linear(len(self.models), 1),
                nn.Sigmoid(),
            )
        elif output_method == "mean":
            # Mean combinator.
            self.predictor = nn.Sequential(
                nn.Flatten(),
                nn.AvgPool1d(len(self.models)),
            )
        elif output_method == "linear":
            # Linear combinator
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
        output = self.predictor(output).squeeze(-1)
        feat_maps.append(output)

        if output_feature_maps:
            return feat_maps
        else:
            return output

    def train_helper(self, _, X, Y, crit, acc_fun, early_acc=1.00):
        chunks = list(zip(torch.tensor_split(X, len(self.models)+1), torch.tensor_split(Y, len(self.models)+1)))[1:]
        loss_overall = 0
        acc = 0

        for i, chunk in enumerate(chunks):
            x, y = chunk[0], chunk[1]
            y_hat = self.models[i](x.clone(), output_feature_maps=False)
            loss = crit(y_hat, y)
            loss_overall += loss
            local_acc = acc_fun(y_hat, y, avg=False)
            acc += local_acc
            if early_acc > (local_acc / len(y)):
                loss.backward()

        return loss_overall / len(self.models), acc / len(self.models)

    def optimize_helper(self, _, X, Y, crit, acc_fun, early_acc=1.00):
        chunks = list(zip(torch.tensor_split(X, len(self.models)+1), torch.tensor_split(Y, len(self.models)+1)))[0]
        x, y = chunks[0], chunks[1]

        for p in self.models.parameters():
            p.requires_grad = False

        y_hat = self.forward(x)
        loss = crit(y_hat, y)
        acc = acc_fun(y_hat, y, avg=False)

        if (self.output_method != "mean") and (early_acc > (acc / len(y))):
            loss.backward()

        return loss, acc
