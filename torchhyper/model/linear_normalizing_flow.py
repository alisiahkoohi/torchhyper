from typing import Tuple

import torch


class LinearNormalizingFlow(torch.nn.Module):
    """Implementation of a downstream network for Gaussian data generation.

    Attributes:
        data_dim: Size of the input data tensor.
    """

    def __init__(self, data_dim: int):
        """Initializes the downstream network weight dimensions.

        Args:
            data_dim: Size of the input data tensor.
        """
        super().__init__()

        self.w = torch.nn.Parameter(torch.randn(data_dim))
        self.b = torch.nn.Parameter(torch.randn(data_dim))

    def forward(
        self,
        x: torch.Tensor,
        rev=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the downstream network.

        Args:
            x: Input tensor of shape (batch_size, data_dim).

        Returns:
            z: A tensor of shape (batch_size, data_dim) with the generated
                latent samples.
            logdet: A tensor of shape (batch_size, 1) with the
                log-determinant of the network Jacobian.
        """
        if not rev:
            # Apply a Linear layer with the given weights and biases.
            z = self.w * x + self.b

            # Compute the log-determinant of the network.
            logdet = torch.sum(torch.log(torch.abs(self.w))).repeat(
                z.shape[0], 1)

            return z, logdet
        else:
            x = self.inverse(x)
            return x

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverts the downstream network.

        Args:
            z: Input tensor of shape (batch_size, data_dim).

        Returns:
            x: A tensor of shape (batch_size, data_dim) with the generated
                data samples.
        """

        # Apply a linear layer with the given weights and biases.
        x = (z - self.b) / (self.w + torch.finfo(self.w.dtype).eps)

        return x
