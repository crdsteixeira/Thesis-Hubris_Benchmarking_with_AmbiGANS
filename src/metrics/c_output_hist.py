import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .metric import Metric
from .hubris import Hubris
from torchvision.transforms import ToTensor
import PIL
import pandas as pd


class OutputsHistogram(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.y_hat = torch.zeros((dataset_size,), dtype=float)
        self.output_clfs = len(C.C.models) if hasattr(C.C, 'models') else 0
        if self.output_clfs:
            self.y_preds = torch.zeros((self.output_clfs, dataset_size,), dtype=float)
            self.to_tensor = ToTensor()
            self.hubris = Hubris(C, dataset_size)

    def update(self, images, batch):
        start_idx, batch_size = batch

        self.hubris.update(images, batch)

        with torch.no_grad():
            c_output, c_all_output = self.C.get(images, start_idx, batch_size, output_feature_maps=True)

        self.y_hat[start_idx:start_idx+batch_size] = c_output
        for i in range(self.output_clfs):
            self.y_preds[i, start_idx:start_idx+batch_size] = c_all_output[0][i]

    def plot(self):
        sns.histplot(data=self.y_hat, stat='proportion', bins=20)

    def plot_clfs(self):
        if not self.output_clfs:
            return None

        clf_hubris = self.hubris.get_clfs()
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        df = pd.DataFrame({
            'y_hat': self.y_hat,
            'cd': torch.abs(0.50 - self.y_hat),
            'Index': ['Ensemble' for _ in range(len(self.y_hat))],
            'Type': ['Ensemble' for _ in range(len(self.y_hat))],
            'sigma': torch.mean(self.y_hat).repeat(len(self.y_hat)),
            'std': torch.std(self.y_hat).repeat(len(self.y_hat)),
            'var': torch.var(self.y_hat).repeat(len(self.y_hat)),
            'hubris': [self.hubris.finalize() for _ in range(len(self.y_hat))],
        })

        for i in range(self.output_clfs):
            df = pd.concat([df, pd.DataFrame({
                'y_hat': self.y_preds[i],
                'cd': torch.abs(0.50 - self.y_preds[i]),
                'Index': [f'CNN_{i}' for _ in range(len(self.y_preds[i]))],
                'Type': ['CNN' for _ in range(len(self.y_hat))],
                'sigma': torch.mean(self.y_preds[i]).repeat(len(self.y_preds[i])),
                'std': torch.std(self.y_preds[i]).repeat(len(self.y_preds[i])),
                'var': torch.var(self.y_preds[i]).repeat(len(self.y_preds[i])),
                'hubris': [clf_hubris[i] for _ in range(len(self.y_hat))],
            })])

        sns.kdeplot(data=df[df['Type'] == "Ensemble"],
                    x='y_hat', hue='Index', alpha=0.5,
                    common_norm=False, legend=False, ax=axs[0, 0])
        axs[0, 0].set(xlim=(0.0, 1.0), title="Ensemble Output Distribution")

        sns.kdeplot(data=df[df['Type'] == "Ensemble"],
                    x='cd', hue='Index', alpha=0.5,
                    common_norm=False, legend=False, ax=axs[1, 0])
        axs[1, 0].set(xlim=(0.0, 1.0), title="Ensemble Output Confusion Distance Distribution")

        sns.kdeplot(data=df[df['Type'] == "CNN"],
                    x='y_hat', hue='Index', alpha=0.5,
                    common_norm=False, legend=False, ax=axs[0, 1])
        axs[0, 1].set(xlim=(0.0, 1.0), title="Individual Classifier Output Distribution")

        sns.kdeplot(data=df[df['Type'] == "CNN"],
                    x='cd', hue='Index', alpha=0.5,
                    common_norm=False, legend=False, ax=axs[1, 1])
        axs[1, 1].set(xlim=(0.0, 1.0), title="Individual Classifier Confusion Distance Distribution")

        sns.scatterplot(data=df,
                        x="sigma", y="std", size="var",
                        hue='Index', alpha=0.5, legend=False, ax=axs[0, 2])
        axs[0, 2].set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), title="Mean vs. Std of Individual Outputs")

        sns.kdeplot(data=df,
                    x="hubris", alpha=0.5,
                    common_norm=False, legend=False, ax=axs[1, 2])
        axs[1, 2].set(xlim=(0.0, 1.0), title="Classifiers' Hubris Distribution")

        # Render and save the picture
        fig.canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return self.to_tensor(pil_image)

    def reset(self):
        pass
