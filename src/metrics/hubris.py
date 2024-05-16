from .metric import Metric
import torch


class Hubris(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.output_clfs = False
        if self.C is not None:
            self.output_clfs = len(self.C.C.models) if hasattr(self.C.C, 'models') else 0
            if self.output_clfs:
                self.clf_preds = torch.zeros((self.output_clfs, dataset_size,), dtype=float)
        self.reset()

    def update(self, images, batch):
        start_idx, batch_size = batch

        with torch.no_grad():
            c_output, c_all_output = self.C.get(images, start_idx, batch_size, output_feature_maps=True)

        self.preds[start_idx:start_idx+batch_size] = c_output
        for i in range(self.output_clfs):
            self.clf_preds[i, start_idx:start_idx+batch_size] = c_all_output[0][i]

    def compute(self, preds, ref_preds=None):
        worst_preds = torch.linspace(0.0, 1.0, steps=preds.size(0))
        binary_preds = torch.hstack((preds, 1.0 - preds))

        if ref_preds is not None:
            reference = ref_preds.clone()
            reference = torch.hstack((reference, 1.0 - reference))
        else:
            reference = torch.full_like(binary_preds, fill_value=0.50)

        binary_worst_preds = torch.hstack((worst_preds, 1.0 - worst_preds))
        predictions_full = torch.distributions.Categorical(probs=binary_preds)
        reference_full = torch.distributions.Categorical(probs=reference)
        worst_preds_full = torch.distributions.Categorical(probs=binary_worst_preds)

        ref_kl = torch.distributions.kl.kl_divergence(worst_preds_full, reference_full).mean()
        amb_kl = torch.distributions.kl.kl_divergence(predictions_full, reference_full).mean()

        return 1.0 - torch.exp(-(amb_kl / ref_kl)).item()

    def finalize(self):
        self.result = self.compute(self.preds)
        return self.result

    def get_clfs(self):
        results = []
        for i in range(self.output_clfs):
            results.append(self.compute(self.clf_preds[i]))
        return results

    def reset(self):
        self.result = torch.tensor([1.0])
        self.preds = torch.zeros((self.dataset_size,), dtype=float)