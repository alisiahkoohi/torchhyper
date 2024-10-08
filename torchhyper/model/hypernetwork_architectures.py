from typing import Dict, List, Optional

import torch
import numpy as np


class HyperNetwork(torch.nn.Module):
    """
    A hypernetwork to predict the weights of a downstream network based on
    input.

    Args:
        input_size (int): The size of the input to the hypernetwork.
        hidden_sizes (List[int]): List of sizes for hidden layers.
        downstream_network (torch.nn.Module): The downstream network whose
        weights need to be predicted.

    Attributes:

        input_size (int): The size of the input to the hypernetwork.
        hidden_sizes (List[int]): List of sizes for hidden layers.
        inner_net (torch.nn.Sequential): Sequential network to process the input
            batch and aggregate the output.
        outer_net (torch.nn.Sequential): Sequential network process the output
            of the inner network.
        target_named_param_sizes (Dict[str, torch.Size]): Dictionary containing
            the names and sizes of the parameters of the downstream network.
        weight_predictors (torch.nn.ModuleList): Module list to predict weights
            of the downstream network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        downstream_network: torch.nn.Module,
        use_outer_net: Optional[bool] = False,
    ) -> None:
        """
        Initializes the HyperNetwork.

        Args:
            input_size (int): The size of the input to the hypernetwork.
            hidden_sizes (List[int]): List of sizes for hidden layers.
            downstream_network (torch.nn.Module): The downstream network whose
                weights need to be predicted.
            use_outer_net (bool): Whether the hypernetwork contains the outer
                network.
        """
        super(HyperNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.use_outer_net = use_outer_net
        self.target_named_param_sizes = {
            k: v.size()
            for k, v in downstream_network.named_parameters()
        }

        # Define the inner network to process the input batch and aggregate the
        # output.
        self.inner_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            self.inner_net.append(
                torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.inner_net.append(torch.nn.ReLU())

        if self.use_outer_net:
            # Define the outer network process the aggregated inner network output.
            self.outer_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                torch.nn.ReLU())
            for _ in range(2):
                self.outer_net.append(
                    torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]))
                self.outer_net.append(torch.nn.ReLU())

        # Define weight predictors
        self.weight_predictors = torch.nn.ModuleList([
            torch.nn.Linear(hidden_sizes[-1], np.prod(param_size))
            for param_size in self.target_named_param_sizes.values()
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hypernetwork.

        Args:
            x (torch.Tensor): Input tensor to the hypernetwork.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the predicted
                weights for each layer of the downstream network.
        """
        # Pass the input batch through the inner_net network and aggregate the
        # output.
        inner_net_outputs = self.inner_net(x.reshape(-1, self.input_size))
        inner_net_outputs = inner_net_outputs.mean(dim=0, keepdim=True)

        if self.use_outer_net:
            # Further process the aggregated inner network outputs.
            inner_net_outputs = self.outer_net(inner_net_outputs)

        # Predict the weights of the downstream network.
        parameters_dict = {
            name: weight.view(param_size)
            for weight, (name, param_size) in zip(
                [layer(inner_net_outputs) for layer in self.weight_predictors],
                self.target_named_param_sizes.items(),
            )
        }

        # Return the parameters_dict.
        return parameters_dict


class NaiveHyperNetwork(torch.nn.Module):
    """
    A hypernetwork to predict the weights of a downstream network based on
    input.

    Args:
        input_size (int): The size of the input to the hypernetwork.
        hidden_sizes (List[int]): List of sizes for hidden layers.
        downstream_network (torch.nn.Module): The downstream network whose
        weights need to be predicted.

    Attributes:

        input_size (int): The size of the input to the hypernetwork.
        hidden_sizes (List[int]): List of sizes for hidden layers.
        inner_net (torch.nn.Sequential): Sequential network to process the input
            batch and aggregate the output.
        outer_net (torch.nn.Sequential): Sequential network process the output
            of the inner network.
        target_named_param_sizes (Dict[str, torch.Size]): Dictionary containing
            the names and sizes of the parameters of the downstream network.
        weight_predictors (torch.nn.ModuleList): Module list to predict weights
            of the downstream network.
    """

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        hidden_sizes: List[int],
        downstream_network: torch.nn.Module,
        use_outer_net: Optional[bool] = False,
    ) -> None:
        """
        Initializes the NaiveHyperNetwork.

        Args:
            batch_size (int): The size of the input batch to the hypernetwork.
            input_size (int): The size of the input to the hypernetwork.
            hidden_sizes (List[int]): List of sizes for hidden layers.
            downstream_network (torch.nn.Module): The downstream network whose
                weights need to be predicted.
            use_outer_net (bool): Whether the hypernetwork contains the outer
                network.
        """
        super(NaiveHyperNetwork, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.use_outer_net = use_outer_net
        self.target_named_param_sizes = {
            k: v.size()
            for k, v in downstream_network.named_parameters()
        }

        # Define the inner network to process the input batch and aggregate the
        # output.
        self.inner_net = torch.nn.Sequential(
            torch.nn.Linear(input_size * self.batch_size, hidden_sizes[0]),
            torch.nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            self.inner_net.append(
                torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.inner_net.append(torch.nn.ReLU())

        if self.use_outer_net:
            # Define the outer network process the aggregated inner network output.
            self.outer_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                torch.nn.ReLU())
            for _ in range(2):
                self.outer_net.append(
                    torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]))
                self.outer_net.append(torch.nn.ReLU())

        # Define weight predictors
        self.weight_predictors = torch.nn.ModuleList([
            torch.nn.Linear(hidden_sizes[-1], np.prod(param_size))
            for param_size in self.target_named_param_sizes.values()
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hypernetwork.

        Args:
            x (torch.Tensor): Input tensor to the hypernetwork.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the predicted
                weights for each layer of the downstream network.
        """
        # Pass the input batch through the inner_net network and aggregate the
        # output.
        inner_net_outputs = self.inner_net(
            x.reshape(-1, self.input_size * self.batch_size))

        if self.use_outer_net:
            # Further process the aggregated inner network outputs.
            inner_net_outputs = self.outer_net(inner_net_outputs)

        # Predict the weights of the downstream network.
        parameters_dict = {
            name: weight.view(param_size)
            for weight, (name, param_size) in zip(
                [layer(inner_net_outputs) for layer in self.weight_predictors],
                self.target_named_param_sizes.items(),
            )
        }

        # Return the parameters_dict.
        return parameters_dict
