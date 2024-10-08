from typing import List

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily,
    kl,
)


class GaussianMixtureModel(torch.nn.Module):
    """Implementation of a downstream network for Gaussian mixture modeling.

    Attributes:
        gmm_dist: torch.distributions.mixture_same_family.MixtureSameFamily:
            Gaussian mixture model data distribution.
    """

    def __init__(
        self,
        input_size: int,
        num_components: int,
        init_temp: float = 1.0,
        min_temp: float = 0.5,
        temp_decay: float = 0.013862944,
    ) -> None:
        """Initializes the Gaussian mixture model.

        Args:
            input_size (int): Dimensionality of the data.
            num_components (int): Number of components in the mixture.
        """
        super().__init__()

        self.init_temp = torch.tensor([init_temp])
        self.min_temp = torch.tensor([min_temp])
        self.temp_decay = temp_decay
        self.gumbel_temp = init_temp
        self.num_components = num_components

        self.mixture_weights = torch.nn.Parameter(torch.randn(num_components))

        # Define mixture means and variances as learnable parameters.
        self.mixture_means = torch.nn.Parameter(
            torch.randn(
                num_components,
                input_size,
            ))
        self.mixture_vars = torch.nn.Parameter(
            torch.randn(
                num_components,
                input_size,
                input_size,
            ))

        self.random_initialization()

    def random_initialization(
            self) -> torch.distributions.mixture_same_family.MixtureSameFamily:
        """
        Intitialize the Gaussian mixture model.
        """

        # Convert the input parameters to tensors.
        mixture_weights: torch.Tensor = torch.rand(self.mixture_weights.size())
        # Normalize the mixture weights.
        mixture_weights = mixture_weights / mixture_weights.sum()

        mixture_means = torch.randn(
            self.mixture_means.size()) * self.num_components**0.5
        mixture_vars = torch.randn(self.mixture_vars.size())
        mixture_vars = mixture_vars @ mixture_vars.transpose(-1, -2)

        self.mixture_weights.data = mixture_weights
        self.mixture_means.data = mixture_means
        self.mixture_vars.data = mixture_vars

    def custom_initialization(
        self,
        mixture_weights: torch.Tensor,
        mixture_means: torch.Tensor,
        mixture_vars: torch.Tensor,
    ) -> torch.distributions.mixture_same_family.MixtureSameFamily:
        """
        Intitialize the Gaussian mixture model.

        Args:
            mixture_weights (torch.Tensor): Weights of the mixture components.
            mixture_means (torch.Tensor): Means of the mixture components.
            mixture_vars (torch.Tensor): Variance of the mixture components.
        """

        # Convert the input parameters to tensors.
        mixture_weights: torch.Tensor = torch.tensor(mixture_weights)
        # Normalize the mixture weights.
        mixture_weights = mixture_weights / mixture_weights.sum()

        mixture_means: List[torch.Tensor] = [
            torch.tensor(mean) for mean in mixture_means
        ]
        mixture_vars: List[torch.Tensor] = [
            torch.eye(len(var)) * torch.tensor(var) for var in mixture_vars
        ]

        self.mixture_weights.data = mixture_weights
        self.mixture_means.data = torch.stack(mixture_means)
        self.mixture_vars.data = torch.stack(mixture_vars)

    def get_mixture_same_family_object(self):
        """
        Get the Gaussian mixture model object.
        """

        return MixtureSameFamily(
            Categorical(self.mixture_weights),
            MultivariateNormal(
                self.mixture_means,
                self.mixture_vars,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        method=None,
        method_args=None,
    ) -> torch.Tensor:
        """Compute the negative log-likelihood of the data samples.

        Args:
            x: Input tensor of shape (batch_size, data_dim).
            method: Method to execute. This is used when other methods need to
                be executed via torch.func.functional_call.
            method_args: Arguments for the method to execute.
        """

        if method == 'sample':
            return self.rsample(method_args[0], method_args[1])

        gmm_dist = self.get_mixture_same_family_object()

        # Compute the negative log-likelihood of the data samples
        return -gmm_dist.log_prob(x)

    def sample(
        self,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample from the Gaussian mixture model.

        Args:
            num_samples (int): Number of samples to generate.
        """
        gmm_dist = MixtureSameFamily(
            Categorical(self.mixture_weights),
            MultivariateNormal(
                self.mixture_means,
                self.mixture_vars,
            ),
        )

        return gmm_dist.sample([num_samples])

    def update_temp(self, epoch: int) -> None:
        """Update the temperature of the Gumbel-Softmax distribution."""

        # Decay gumbel temperature
        if self.temp_decay > 0:
            decay_factor = torch.exp(-self.temp_decay * epoch)
            self.gumbel_temp = torch.max(self.init_temp * decay_factor,
                                         self.min_temp)

    def rsample(
        self,
        num_samples: int,
        epoch,
    ) -> torch.Tensor:
        """Sample from the Gaussian mixture model.

        Args:
            num_samples (int): Number of samples to generate.
            epoch (int): Current epoch.
        """

        # Choose which component to sample from based on `mixture_weights`. Sample
        # with reparameterization trick from the Gumber-Softmax distributuion so
        # that gradients can flow.
        self.update_temp(epoch)
        component_idxs = torch.nn.functional.gumbel_softmax(
            self.mixture_weights.repeat(num_samples, 1),
            tau=self.gumbel_temp.to(self.mixture_weights.device),
            hard=False,
        )

        # Sample from the chosen component
        sampled_means = self.mixture_means[torch.argmax(component_idxs, dim=1)]
        sampled_vars = self.mixture_vars[torch.argmax(component_idxs, dim=1)]

        # Generate samples using broadcasting.
        samples = MultivariateNormal(sampled_means.unsqueeze(1),
                                     sampled_vars.unsqueeze(1)).rsample()

        return samples[:, 0, :]


@kl.register_kl(MixtureSameFamily, MixtureSameFamily)
def kl_gmm_gmm(p, q, n_samples=10**5):
    """
    Computes an approximation of the KL divergence between two Gaussian Mixture Models (GMMs) using Monte Carlo sampling.

    Args:
        p (MixtureSameFamily): The first GMM distribution.
        q (MixtureSameFamily): The second GMM distribution.
        n_samples (int): The number of samples to use for Monte Carlo estimation.

    Returns:
        Tensor: An approximation of the KL divergence between the two GMMs.
    """
    x = p.sample([n_samples])
    log_p_x = p.log_prob(x)
    log_q_x = q.log_prob(x)
    kl_div = log_p_x.mean() - log_q_x.mean()
    return kl_div
