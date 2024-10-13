# `torchhyper`: A PyTorch library for modular hypernetworks

## Installation

Run the command below to install the package to be used in your Python environment.

```bash
pip install torchhyper
```

For further development and to run the examples, clone the repository
and install the package in editable mode. **Make sure to adapt CUDA
version in `setup.cfg` to the one installed on your system.**

```bash
# Create a new conda environment.
conda create --name torchhyper "python<=3.12"
conda activate torchhyper

# Clone the repository and install the package in editable mode.
git clone ttps://github.com/alisiahkoohi/torchhyper
cd torchhyper/
pip install -e .
```

## Usage

The hypernetwork module `HyperNetwork`, found in
`torchhyper/models/architecture.py`, should have the ability to adapt itself to the
architecture of any given downstream network. This means that the only
modifications required in any given script to train the hypernetwork for
generating weights for a specific downstream network are outlined below.
First, define the hypernetwork and its optimizer as follows:

```python
from torchhyper.model import HyperNetwork

# Define your downstream network, e.g., a torch.nn.Module instance for a diffusion model.
downstream_network = YourDownstreamNetwork() # Any torch.nn.Module instance.

# No need to train the downstream network directly.
downstream_network.requires_grad_(False)

# Define the hypernetwork.
hypernetwork = HyperNetwork(
    input_dim,      # e.g., x[0, ...].numel() for input variable x (excluding batch dimension).
    [32, 64, 96], # A list of hidden layer sizes of the hypernetwork layers.
    downstream_net,
)

# Define the optimizer to train the hypernetwork.
optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=1e-3)
```

Next, replace every instance of `downstream_network` forward evaluation
`downstream_network(x)` for input tensor `x` with the following functional call
to the downstream network using the predicted weights by the hypernetwork:

```python
# Predicting the weight dictionary for the downstream network.
weight_dict = hypernetwork(x)

# Predicting the output of the downstream network using the predicted weights.
pred = torch.func.functional_call(
    downstream_network, # The downstream network.
    weight_dict,        # The predicted weight dictionary.
    x,                  # The input to the downstream network.
)
```

The downstream network output `pred` is now equivalent to
`downstream_network(x)` using the predicted weights by the hypernetwork
and can be similarly utilized for loss computation and backpropagation.

An simple test case for the gradient calculation with this approach can
be found in `tests/test_functional_call.py`.

When calling other methods of the downstream network, e.g.,
`downstream_network.sample()`, that do not require gradient calculation,
the predicted weights by the hypernetwork can be directly passed to the
downstream network as follows before calling the method:

```python
# Predicting the weight dictionary for the downstream network.
weight_dict = hypernetwork(x)

# Set the parameters of the downstream network to the computed ones.
for param, weight in zip(
        downstream_network.parameters(),
        weight_dict.values(),
):
    param.data = weight.data

# Calling the `sample` method of the downstream network (no need for gradient calculation).
downstream_network.sample()
```

## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi




