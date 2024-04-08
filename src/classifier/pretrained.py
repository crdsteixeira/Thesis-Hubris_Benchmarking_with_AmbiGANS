import torch.nn as nn
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
from torchvision import transforms

class ClassifierVIT(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(ClassifierVIT, self).__init__()
        num_channels = img_size[0]
        out_features = 1 if num_classes == 2 else num_classes
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        self.model.num_labels = out_features
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def forward(self, x, output_feature_maps=False):
        images = torch.stack([self.transforms(i).to(self.device) for i in x])
        return self.model(images)['logits']

class ClassifierResnet(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(ClassifierResnet, self).__init__()
        num_channels = img_size[0]
        out_features = 1 if num_classes == 2 else num_classes
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-mnist")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.model.config.hidden_sizes[-1], out_features=out_features),
        )
        self.model.num_labels = out_features

    def forward(self, x, output_feature_maps=False):
        return self.model(x)['logits']

class ClassifierMLP(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(ClassifierMLP, self).__init__()
        num_channels = img_size[0]
        out_features = 1 if num_classes == 2 else num_classes
        self.device = device
        self.model = AutoModel.from_pretrained("dacorvo/mnist-mlp", trust_remote_code=True)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.model.config.hidden_size, out_features=out_features),
        )
        self.transforms = transforms.Compose([
            transforms.Normalize((0.1307), (0.3081)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])

    def forward(self, x, output_feature_maps=False):
        images = torch.stack([self.transforms(i).to(self.device) for i in x])
        return self.model(images)

class Ensemble(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(Ensemble, self).__init__()
        num_channels = img_size[0]
        self.device = device
        #self.model_1 = ClassifierVIT(img_size, num_classes, nf, self.device)
        self.model_2 = ClassifierResnet(img_size, num_classes, nf, self.device)
        self.model_3 = ClassifierMLP(img_size, num_classes, nf, self.device)
        self.models = [
            self.model_2,
            self.model_3,
        ]
        # TO DISCUSS
        self.predictor = nn.Sequential(
            nn.Linear(len(self.models), 1),
            nn.Sigmoid(),
        )
        # TO DISCUSS

    def forward(self, x, output_feature_maps=False):
        output = torch.Tensor().to(self.device)
        feat_maps = []
        for m in self.models:
            out = m(x.clone(), output_feature_maps)
            output = torch.cat((output, out), dim=1)

        feat_maps.append(output)
        output = self.predictor(output).squeeze(1)
        feat_maps.append(output)

        if output_feature_maps:
            return feat_maps
        else:
            return output