import torch.autograd as autograd
from src.utils.min_norm_solvers import MinNormSolver
import numpy as np
from torch.nn.utils import clip_grad_norm_ 
from torch.nn import GaussianNLLLoss, KLDivLoss, BCELoss
from torch import full_like, hstack, log, randn_like


class UpdateGenerator:
    def __init__(self, crit):
        self.crit = crit

    def __call__(self, G, D, optim, noise, device):
        raise NotImplementedError

    def get_loss_terms(self):
        raise NotImplementedError


class UpdateGeneratorGAN(UpdateGenerator):
    def __init__(self, crit):
        super().__init__(crit)

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()

        fake_data = G(noise)
        output = D(fake_data)

        loss = self.crit(device, output)

        loss.backward()
        optim.step()

        return loss, {}

    def get_loss_terms(self):
        return []


class UpdateGeneratorGASTEN(UpdateGenerator):
    def __init__(self, crit, C, alpha):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()

        fake_data = G(noise)

        output = D(fake_data)
        term_1 = self.crit(device, output)

        clf_output = self.C(fake_data)
        term_2 = (0.5 - clf_output).abs().mean()

        loss = term_1 + self.alpha * term_2

        loss.backward()
        optim.step()

        return loss, {'original_g_loss': term_1.item(), 'conf_dist_loss': term_2.item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']


class UpdateGeneratorGASTEN_MGDA(UpdateGenerator):
    def __init__(self, crit, C, alpha=1, normalize=False):
        super().__init__(crit)
        self.C = C
        self.alpha = 1
        self.normalize = normalize

    def gradient_normalizers(self, grads, loss):
        return loss.item() * np.sqrt(np.sum([gr.pow(2).sum().data.cpu()
                                             for gr in grads]))

    def __call__(self, G, D, optim, noise, device):
        # Compute gradients of each loss function wrt parameters

        # Term 1
        G.zero_grad()

        fake_data = G(noise)

        output = D(fake_data)
        term_1 = self.crit(device, output)

        term_1.backward()
        term_1_grads = []
        for param in G.parameters():
            if param.grad is not None:
                term_1_grads.append(
                    autograd.Variable(param.grad.data.clone(), requires_grad=False))

        # Term 2
        G.zero_grad()

        fake_data = G(noise)

        c_output = self.C(fake_data)
        term_2 = (0.5 - c_output).abs().mean()

        term_2.backward()
        term_2_grads = []
        for param in G.parameters():
            if param.grad is not None:
                term_2_grads.append(
                    autograd.Variable(param.grad.data.clone(), requires_grad=False))

        if self.normalize:
            gn1 = self.gradient_normalizers(term_1_grads, term_1)
            gn2 = self.gradient_normalizers(term_2_grads, term_2)

            for gr_i in range(len(term_1_grads)):
                term_1_grads[gr_i] = term_1_grads[gr_i] / gn1
            for gr_i in range(len(term_2_grads)):
                term_2_grads[gr_i] = term_2_grads[gr_i] / gn2

        scale, min_norm = MinNormSolver.find_min_norm_element(
            [term_1_grads, term_2_grads])

        # Scaled back-propagation
        G.zero_grad()

        fake_data = G(noise)

        output = D(fake_data)
        term_1 = self.crit(device, output)

        clf_output = self.C(fake_data)
        term_2 = (0.5 - clf_output).abs().mean()

        loss = scale[0] * term_1 + scale[1] * term_2

        loss.backward()
        optim.step()

        return loss, {'original_g_loss': term_1.item(), 'conf_dist_loss': term_2.item(), 'scale1': scale[0], 'scale2': scale[1]}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss', 'scale1', 'scale2']

class UpdateGeneratorGASTEN_gaussian(UpdateGenerator):
    def __init__(self, crit, C, alpha, var):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss()
        self.target = 0.5

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()

        optim.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data)
        # update from ensemble
        target = full_like(input=clf_output, fill_value=self.target, device=device)
        var = full_like(input=clf_output, fill_value=self.var, device=device)
        loss_1 = self.c_loss(clf_output, target, var)
        loss_1.backward()
        clip_grad_norm_(G.parameters(), 0.50 * self.alpha)
        optim.step()
        # update from discriminator
        optim.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        loss_2 = self.crit(device, output)
        loss_2.backward()
        clip_grad_norm_(G.parameters(), 0.50)
        optim.step()

        loss = loss_1 + loss_2

        return loss, {'original_g_loss': loss_2.item(), 'conf_dist_loss': loss_1.item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']

class UpdateGeneratorGASTEN_KLDiv(UpdateGenerator):
    def __init__(self, crit, C, alpha):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.c_loss = KLDivLoss(reduction="none")
        self.target = 0.5
        self.eps = 1e-9
        self.crit = BCELoss(reduction="none")
            
    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()

        optim.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data, output_feature_maps=True)
        # update from ensemble
        loss_1 = 0
        for c_pred in clf_output[0].T:
            class_prob = hstack((c_pred.unsqueeze(-1), 1.0-c_pred.unsqueeze(-1)))
            target = full_like(input=class_prob, fill_value=self.target, device=device)
            loss_1 += self.c_loss(log(class_prob.clip(self.eps, 1.0)), target)
        # update from discriminator
        output = D(fake_data)
        target = full_like(input=output, fill_value=1.0, device=device)
        loss_2 = self.crit(output, target)
        loss = hstack((self.alpha * loss_1, loss_2.unsqueeze(-1))).sum()
        loss.backward()
        optim.step()

        return loss, {'original_g_loss': loss_2.sum().item(), 'conf_dist_loss': loss_1.sum().item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']

class UpdateGeneratorGASTEN_gaussianV2(UpdateGenerator):
    def __init__(self, crit, C, alpha, var):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss(reduction="none")
        self.target = 0.5
        self.crit = BCELoss(reduction="none")

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()

        optim.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data, output_feature_maps=True)
        # update from ensemble
        loss_1 = 0
        for c_pred in clf_output[0].T:
            target = full_like(input=c_pred, fill_value=self.target, device=device)
            var = full_like(input=c_pred, fill_value=self.var, device=device)
            loss_1 += self.c_loss(c_pred, target, var)
        # update from discriminator
        output = D(fake_data)
        target = full_like(input=output, fill_value=1.0, device=device)
        loss_2 = self.crit(output, target)
        loss = hstack((self.alpha * loss_1, loss_2)).sum()
        loss.backward()
        optim.step()

        return loss, {'original_g_loss': loss_2.sum().item(), 'conf_dist_loss': loss_1.sum().item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']