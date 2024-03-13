import torch.nn as nn
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms

class ClassifierVIT(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(ClassifierVIT, self).__init__()
        num_channels = img_size[0]
        self.device = device
        self.model = model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")
        self.dropout = nn.Dropout(0.10)
        self.predictor = nn.Sequential(
            nn.Linear(10, 1 if num_classes ==
                      2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
        )
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def forward(self, x, output_feature_maps=False):
        output = torch.Tensor().to(self.device)
        for i in x:
            o = self.model(self.transforms(i).unsqueeze(0).to(self.device))
            out = self.dropout(o['logits']) # Pick up logits
            output = torch.cat((output, out), dim=0)

        if output_feature_maps:
            return output
        else:
            return self.predictor(output)

class Ensemble(nn.Module):
    def __init__(self, img_size, num_classes, nf, device):
        super(Ensemble, self).__init__()
        num_channels = img_size[0]
        self.device = device
        self.models = [
            ClassifierVIT(img_size, num_classes, nf, self.device).to(self.device),
        ]
        # TO DISCUSS
        self.predictor = nn.Sequential(
            nn.Linear(len(self.models), 1),
        ).to(self.device)
        # TO DISCUSS

    def forward(self, x, output_feature_maps=False):
        output = torch.Tensor().to(self.device)
        for m in self.models:
            out = m(x.to(self.device), output_feature_maps)
            output = torch.cat((output, out), dim=1)

        return self.predictor(output).squeeze(1)