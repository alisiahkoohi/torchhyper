import torch
from torch.nn import functional as F


class Flatten(torch.nn.Module):
    """
    Flatten layer: Flattens the input tensor.

    Args:
        None

    Returns:
        torch.Tensor: Flattened tensor
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method of Flatten layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Flattened tensor
        """
        return x.view(x.size(0), -1)


class Reshape(torch.nn.Module):
    """
    Reshape layer: Reshapes the input tensor according to the specified shape.

    Args:
        outer_shape (tuple): Target shape for reshaping

    Returns:
        torch.Tensor: Reshaped tensor
    """

    def __init__(self, outer_shape: tuple):
        """
        Initialize Reshape layer.

        Args:
            outer_shape (tuple): Target shape for reshaping
        """
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method of Reshape layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reshaped tensor
        """
        return x.view(x.size(0), *self.outer_shape)


class Gaussian(torch.nn.Module):
    """
    Gaussian layer: Generates samples from a Gaussian distribution.

    Args:
        in_dim (int): Input dimension
        z_dim (int): Latent space dimension

    Returns:
        tuple: Tuple containing mean, variance, and latent space vector
    """

    def __init__(self, in_dim: int, z_dim: int):
        """
        Initialize Gaussian layer.

        Args:
            in_dim (int): Input dimension
            z_dim (int): Latent space dimension
        """
        super(Gaussian, self).__init__()
        self.mu = torch.nn.Linear(in_dim, z_dim, bias=False)
        self.var = torch.nn.Linear(in_dim, z_dim, bias=False)

    def reparameterize(self, mu: torch.Tensor,
                       var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for Gaussian sampling.

        Args:
            mu (torch.Tensor): Mean tensor
            var (torch.Tensor): Variance tensor

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass method of Gaussian layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Tuple containing mean, variance, and latent space vector
        """
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class SkipConnection(torch.nn.Module):
    """
    Skip Connection layer: Adds the input tensor to the output of another module.

    Args:
        module (torch.nn.Module): Module whose output will be added to the input

    Returns:
        torch.Tensor: Output tensor with skip connection
    """

    def __init__(self, module: torch.nn.Module):
        """
        Initialize Skip Connection layer.

        Args:
            module (torch.nn.Module): Module whose output will be added to the input
        """
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method of Skip Connection layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with skip connection
        """
        return x + self.module(x)


class View(torch.nn.Module):
    """
    A module to create a view of an existing torch.Tensor (avoids copying).
    Attributes:
        shape: A tuple containing the desired shape of the view.
    """

    def __init__(self, *shape: int) -> None:
        """
        Initializes a Concat module.

        Args:
            shape (Tuple[int]): A tuple containing the desired shape of the
                view.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates a view of an input.
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, *input_shape).
        Returns:
            torch.Tensor: A tensor of shape (batch_size, *self.shape).
        """
        # Use the `view` method of the input tensor to create a view with the
        # desired shape.
        return x.view(*self.shape)
