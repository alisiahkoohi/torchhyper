import torch
from torch.nn import functional as F


class LossFunctions(object):
    """Loss functions for the GMVAE."""

    eps = 1e-8

    def reconstruction_loss(self, real, predicted, rec_type='mse'):
        """Reconstruction loss between the true and predicted outputs
        mse = (1/n)*Σ(real - predicted)^2
        bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
        """
        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise ValueError("invalid loss function... try bce or mse...")
        return loss.sum(tuple(range(1, loss.dim()))).mean()

    def exact_gaussian_loss(self, z_mu, z_var):
        """Variational loss when using labeled data without considering reconstruction loss
        loss = log q(z|x) - log p(z)

        Args:
            z: (array) array containing the gaussian latent variable
            z_mu: (array) array containing the mean of the inference model
            z_var: (array) array containing the variance of the inference model

        Returns:
            output: (array/float) depending on average parameters the result will be the mean  of all the sample losses or an array with the losses per sample
        """

        loss = -0.5 * z_mu.shape[-1]
        loss += -0.5 * torch.sum(torch.log(z_var), dim=-1)
        loss += 0.5 * torch.sum(z_var + torch.pow(z_mu, 2), dim=-1)
        return loss.mean()

    def gaussian_loss(self, z, z_mu, z_var):
        """Variational loss when using labeled data without considering reconstruction loss
        loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
            z: (array) array containing the gaussian latent variable
            z_mu: (array) array containing the mean of the inference model
            z_var: (array) array containing the variance of the inference model
            z_mu_prior: (array) array containing the prior mean of the generative model
            z_var_prior: (array) array containing the prior variance of the generative mode

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                of all the sample losses or an array with the losses per sample
        """

        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(
            z, torch.zeros_like(z), torch.ones_like(z))
        return loss.mean()
