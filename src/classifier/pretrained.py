import torch.nn as nn
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms

class ClassifierVIT(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(ClassifierVIT, self).__init__()
        num_channels = img_size[0]
        out_features = 1 if num_classes == 2 else num_classes
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")
        self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features, bias=True)
        self.model.num_labels = out_features
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def forward(self, x, output_feature_maps=False):
        output = torch.Tensor().to(self.device)
        images = torch.stack([self.transforms(i) for i in x]).to(self.device)
        return self.model(images)['logits']

class Ensemble(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(Ensemble, self).__init__()
        num_channels = img_size[0]
        self.device = device
        self.model_1 = ClassifierVIT(img_size, num_classes, nf, self.device).to(self.device)
        self.models = [
            self.model_1,
        ]
        # TO DISCUSS
        self.predictor = nn.Sequential(
            nn.Linear(len(self.models), 1),
            nn.Sigmoid(),
        ).to(self.device)
        # TO DISCUSS

    def forward(self, x, output_feature_maps=False):
        output = torch.Tensor().to(self.device)
        feat_maps = []
        for m in self.models:
            out = m(x.to(self.device), output_feature_maps)
            output = torch.cat((output, out), dim=1)

        feat_maps.append(output)
        output = self.predictor(output).squeeze(1)
        feat_maps.append(output)

        if output_feature_maps:
            return feat_maps
        else:
            return output