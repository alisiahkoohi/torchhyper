# pylint: disable=E1102
import unittest
from typing import Dict

import torch


class TestGradients(unittest.TestCase):
    """Test case for gradient computation for hypernetworks.

    Initializing the weights of a downstream network using the output of a
    hypernetwork is non-trivial. By simply using the output of the hypernetwork
    to update the weights of the downstream network, the computational graph is
    broken and the gradients with respect to the hypernetwork weights are not
    computed.

    An alternative is to use `torch.nn.functional` to reimplement the forward
    pass of the downstream network. This way, the computational graph is
    preserved and the gradients with respect to the hypernetwork weights are
    computed. However, this requires reimplementing from scratch every new
    downstream network using `torch.nn.functional` equivalent functions.

    Here, we use `torch.func.functional_call` to evaluate the forward pass of
    the downstream network when using the output of the hypernetwork as weights
    while also preserving the computational graph and computing the gradients
    with respect to the hypernetwork weights. This allows any downstream network
    to be used without reimplementing it using
    `torch.nn.functional`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the test case."""
        super(TestGradients, self).__init__(*args, **kwargs)

        # Generate random tensor of shape (10, 10) as dummy input data.
        self.x = torch.rand(10, 10)

        # Dummy linear layer without bias as hypernetwork.
        self.hyper_net = torch.nn.Linear(10, 10, bias=False)

        # Dummy linear layer without bias as downstream network.
        self.downstream_net = torch.nn.Linear(10, 10, bias=False)

        # Compute the output of the hypernetwork to be used as weights of the
        # downstream network.
        self.A = self.hyper_net(self.x)

        # Creat a dictionary of the parameters using the output of the
        # hypernetwork that has the same structure as the named parameters
        # dictionary of the downstream network.
        self.parameter_dicts: Dict[str, torch.Tensor] = {
            k: pv.reshape(v.shape)
            for (k, v), pv in zip(
                dict(self.downstream_net.named_parameters()).items(),
                (self.A,),
            )
        }

    def test_forward_pass(self) -> None:
        """Test downstream network output."""

        # Compute output using functional.linear.
        y_functional = torch.nn.functional.linear(self.x, self.A)

        # Compute output using functional_call.
        y_functional_call = torch.func.functional_call(
            self.downstream_net,
            self.parameter_dicts,
            self.x,
        )

        # Assert if y is close to y2
        self.assertTrue(torch.allclose(y_functional_call, y_functional))

    def test_gradient(self) -> None:
        """Test gradient computation."""

        # Compute output using functional.linear.
        y_functional = torch.nn.functional.linear(self.x, self.A)

        # Compute gradients for functional.linear.bin
        y_functional.sum().backward(retain_graph=True)

        # Save gradients of hyper_net
        grad_functional = self.hyper_net.weight.grad.clone()

        # Zero gradients for hyper_net
        self.hyper_net.zero_grad()

        # Compute output using functional_call.
        y_functional_call = torch.func.functional_call(
            self.downstream_net,
            self.parameter_dicts,
            self.x,
        )
        y_functional_call.sum().backward(retain_graph=True)
        # Get gradients of hyper_net after backward pass
        grad_functional_call = self.hyper_net.weight.grad

        # Check if gradients are close with the two methods.
        self.assertTrue(torch.allclose(grad_functional_call, grad_functional))


if __name__ == "__main__":
    unittest.main()
