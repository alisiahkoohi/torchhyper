# pylint: disable=E1102
# pylint: disable=invalid-name
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


def setup_gaussian_dist(
    data_dim: int,
    device: torch.device,
) -> MultivariateNormal:
    """
    Set up Gaussian distribution.

    Args:
        data_dim (int): Dimensionality of the data.
        device (torch.device): Device on which the tensors are created.

    Returns:
        torch.distributions.MultivariateNormal: Gaussian data distribution.
    """
    # Data distribution.
    mu_x: torch.Tensor = torch.ones(data_dim, device=device)  # Data mean.
    cov_x: torch.Tensor = torch.from_numpy(
        np.diag(np.arange(1, data_dim + 1)).astype(np.float32)).to(
            device)  # Data covariance matrix.

    # Initialize and return data distribution.
    return MultivariateNormal(mu_x, covariance_matrix=cov_x)
