from typing import Tuple, Dict

import torch

from .layers import Gaussian


class Encoder(torch.nn.Module):
    """
    A module to create an encoder network for the VAE.
    """

    def __init__(
        self,
        x_shape: int,
        z_dim: int,
        hidden_dim: int,
        nlayer: int,
    ) -> None:
        """
        Initializes the Encoder module.

        Args:
            x_shape (int): Integer representing the shape of flattened input.
            z_dim (int): Dimensionality of the latent variable z.
            hidden_dim (int): Dimensionality of the hidden layer.
            nlayer (int): Number of layers in the neural network.

        Returns:
            None
        """
        super(Encoder, self).__init__()

        # q(z|x)
        self.inference_qyx = torch.nn.ModuleList([
            # Add a linear layer to the current ModuleList with
            # input dimensions of x_shape and output dimensions
            # of hidden_dim.
            torch.nn.Linear(x_shape, hidden_dim, bias=False),
            # Add a batch normalization layer to the current ModuleList
            # with input dimensions of hidden_dim
            torch.nn.BatchNorm1d(hidden_dim),
            # Add a LeakyReLU activation layer to the current ModuleList
            torch.nn.LeakyReLU(negative_slope=0.2)
        ])

        # Add more layers to the q(z|x) ModuleList.
        for _ in range(1, nlayer):
            # Add a linear layer to the current ModuleList with
            # input and output dimensions of hidden_dim
            self.inference_qyx.append(
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
            # Add a batch normalization layer to the current ModuleList
            # with input dimensions of hidden_dim
            self.inference_qyx.append(torch.nn.BatchNorm1d(hidden_dim))
            # Add a LeakyReLU activation layer to the current ModuleList
            self.inference_qyx.append(torch.nn.LeakyReLU(negative_slope=0.2))
        # Create a sequential module from the layers added to the ModuleList
        self.inference_qyx = torch.nn.Sequential(*self.inference_qyx)

        # Create a Gaussian module.
        self.gaussian = Gaussian(hidden_dim, z_dim)

    def qzx(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the approximate posterior distribution q(z|x).

        Args:
            x (torch.Tensor): Tensor representing the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple representing the mean and
                logvariance of the Gaussian distribution.
        """
        # Pass the input tensors through the q(z|x) network
        x = self.inference_qyx(x)
        # Compute the mean and logvariance of the Gaussian distribution
        x = self.gaussian(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the forward pass of the model.

        Args:
            x (torch.Tensor): Tensor representing the input.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the output tensors.
                - 'mean': Tensor representing the mean tensors of the Gaussian
                      distributions.
                - 'var': Tensor representing the variance tensors of the
                      Gaussian distributions.
                - 'gaussian': Tensor representing the latent variable tensors
                      sampled from the Gaussian distributions.
        """
        # q(z|x)
        x = x.view(x.size(0), -1)
        mu, var, z = self.qzx(x)

        return {
            'mean': mu,
            'var': var,
            'gaussian': z,
        }


class Decoder(torch.nn.Module):
    """
    A module to create a decoder network for the VAE.
    """

    def __init__(
        self,
        x_shape: int,
        z_dim: int,
        hidden_dim: int,
        nlayer: int,
    ) -> None:
        """
        Decoder network that generates an image given the latent variables z.

        Args:
            x_shape (int): An integer representing data length.
            z_dim (int): Dimensionality of the latent variable z.
            hidden_dim (int): Dimensionality of the hidden layers.
            nlayer (int): Number of hidden layers.
        """
        super(Decoder, self).__init__()

        # Define the p(x|z) generative network architecture.
        self.generative_pxz = torch.nn.ModuleList([
            # First layer: linear transformation from z_dim to hidden_dim.
            torch.nn.Linear(z_dim, hidden_dim, bias=False),
            # Batch normalization layer.
            torch.nn.BatchNorm1d(hidden_dim),
            # Activation function.
            torch.nn.LeakyReLU(negative_slope=0.2),
        ])

        # Add additional layers to the generative network architecture.
        for _ in range(1, nlayer):
            self.generative_pxz.append(
                # Additional linear layer with hidden_dim input and output
                # dimensions.
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.generative_pxz.append(
                # Additional batch normalization layer.
                torch.nn.BatchNorm1d(hidden_dim))
            self.generative_pxz.append(
                # Additional activation function.
                torch.nn.LeakyReLU(negative_slope=0.2))
        self.generative_pxz.append(
            # Final linear layer with hidden_dim input dimension and
            # x_shape output dimension.
            torch.nn.Linear(hidden_dim, x_shape))
        self.generative_pxz = torch.nn.Sequential(
            # Create a sequential model by combining all layers of the
            # generative network architecture
            *self.generative_pxz)

    def forward(
        self,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the output of the model given the latent variable.

        Args:
            z (torch.Tensor): A tensor representing the latent variable.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the model output.
        """
        # Compute the probability distribution of the data given the latent variable.
        x = self.generative_pxz(z)

        # Return the model output.
        return {'x_rec': x}


class VAE(torch.nn.Module):
    """
    A module implementing a plain variational autoencoder (VAE).

    Args:
        x_shape (tuple): The shape of the input data.
        z_dim (int): The dimensionality of the continuous latent variable.
        hidden_dim (int): The dimensionality of the hidden layers.
        nlayer (int): The number of hidden layers.

    Attributes:
        inference (Encoder): The encoder network.
        generative (Decoder): The decoder network.
    """

    def __init__(
        self,
        x_shape: Tuple[int, int, int],
        z_dim: int,
        hidden_dim: int = 512,
        nlayer: int = 3,
    ) -> None:
        super(VAE, self).__init__()

        self.z_dim = z_dim

        # Instantiate the encoder and decoder networks.
        self.inference = Encoder(x_shape, z_dim, hidden_dim, nlayer)
        self.generative = Decoder(x_shape, z_dim, hidden_dim, nlayer)

    def forward(
        self,
        x: torch.Tensor,
        method=None,
        method_args=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): The input data.
            method (str): The method to use for the forward pass.
            method_args (list): The arguments to pass to the method.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the output of the
                inference and generative networks.
        """

        # If the method is 'sample', generate samples from the VAE. This is used
        # when evaluating the network using torch.func.functional to allow for
        # backpropagation through sampling.
        if method == 'sample':
            return self.sample(method_args[0])

        x = x.view(x.size(0), -1)
        # Pass the input data through the encoder network.
        out_inf = self.inference(x)

        # Extract the latent variable from the output of the encoder network.
        z = out_inf['gaussian']

        # Pass the latent variables through the decoder network.
        out_gen = self.generative(z)

        # Combine the output of the encoder and decoder networks.
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the VAE.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated samples.
        """
        # Sample from the standard normal distribution.
        z = torch.randn(
            num_samples,
            self.z_dim,
            device=list(self.generative.parameters())[0].device,
        )

        # Pass the samples through the decoder network.
        samples = self.generative(z)['x_rec']

        return samples
