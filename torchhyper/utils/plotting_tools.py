import os
from typing import List, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from .project_path import plotsdir
from ..model import GaussianMixtureModel

sns.set_style("whitegrid")
font: dict[str, Any] = {'family': 'serif', 'style': 'normal', 'size': 14}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")
np.set_printoptions(precision=4)


def plot_loss(
    args: Any,
    train_loss: List[float],
    val_loss: Optional[List[float]] = [],
    epoch=-1,
    log_scale: bool = False,
    filename_ext: str = "",
    label_ext: str = "",
) -> None:
    """
    Plot training loss over training iterations and save the plot.

    Parameters:
        args (Any): Arguments for the experiment.
        train_loss (List[float]): List of training losses over iterations.
        val_loss (List[float]): Optional list of training losses over iterations.
        epoch (int): The epoch corresponding to the data.
        log_scale (bool): Whether to plot the y-axis in log scale.
        filename_ext (str): The filename extension.
        label_ext (str): The label extension.

    Returns:
        None
    """

    if epoch == -1:
        epoch = args.testing_epoch

    # Create a new figure for plotting
    fig = plt.figure("training logs", figsize=(7, 4))

    # Plot training (and validation) loss
    plt.plot(
        np.linspace(0, epoch + 1,
                    (epoch + 1) * len(train_loss) // (epoch + 1)),
        train_loss,
        color="orange",
        alpha=1.0,
        label="training loss" + label_ext,
    )
    if val_loss:
        plt.plot(
            np.linspace(0, epoch + 1, epoch + 1),
            val_loss,
            color="k",
            alpha=0.8,
            label="validation loss" + label_ext,
        )

    # Format y-axis labels using scientific notation
    plt.ticklabel_format(axis="y", style="sci", useMathText=True)

    # Set plot title and labels
    plt.title("Training loss over training")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")

    # Save the plot
    plt.savefig(os.path.join(
        plotsdir(args.experiment),
        "log" + filename_ext + ".png",
    ),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)

    # Close the figure to release memory
    plt.close(fig)


def plot_gaussian_example(
    args: Any,
    x: torch.Tensor,
    x_val: torch.Tensor,
    train_loss: List[float],
    val_loss: List[float],
    data_dist: torch.distributions.Distribution,
) -> None:
    """
    Plot Gaussian examtorchhyper.

    Args:
        args (Any): Arguments for plotting.
        x (torch.Tensor): Training data tensor.
        x_val (torch.Tensor): Validation data tensor.
        train_loss (List[float]): Training loss values.
        val_loss (List[float]): Validation loss values.
        data_dist (torch.distributions.Distribution): Distribution of the data.
    """

    # Plot training loss.
    plot_loss(args, train_loss, val_loss=val_loss)

    # Print the true mean and std and compare with the estimated mean and std.
    true_mean = data_dist.mean.cpu().numpy()
    estimated_mean = x.mean(dim=0).cpu().numpy()
    true_variance = data_dist.covariance_matrix.diag().cpu().numpy()
    estimated_variance = x.std(dim=0).cpu().numpy()**2

    print("\nTrue mean:       ", true_mean)
    print("Estimated mean:  ", estimated_mean, "\n")
    print("True variance:   ", true_variance)
    print("Estimated variance: ", estimated_variance, "\n")

    # True samples
    fig = plt.figure(figsize=(7, 5))
    plt.plot(
        x_val[:1000, :].T.cpu().numpy(),
        "*",
        linewidth=0.8,
        alpha=0.3,
        color="k",
    )
    plt.xlim([-1, args.data_dim])
    plt.ylim([-args.data_dim * 1.2, args.data_dim * 1.2])
    plt.grid(True)
    plt.title(r"$\mathbf{x} \sim p(\mathbf{x})$")
    plt.savefig(
        os.path.join(plotsdir(args.experiment), "true_samples.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    # Estimated samples
    fig = plt.figure(figsize=(7, 5))
    plt.plot(
        x[:1000, :].T.cpu().numpy(),
        "*",
        linewidth=0.8,
        alpha=0.3,
        color="#22c1d6",
    )
    plt.xlim([-1, args.data_dim])
    plt.ylim([-args.data_dim * 1.2, args.data_dim * 1.2])
    plt.grid(True)
    plt.title(r"$\mathbf{x} \sim p_{\theta} (\mathbf{x})$")
    plt.savefig(
        os.path.join(
            plotsdir(args.experiment),
            "estimated_samples.png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_gmm_example(
    args: Any,
    train_loss: List[float],
    val_loss: List[float],
    data_dist: torch.distributions.mixture_same_family.MixtureSameFamily,
    gmm_model: torch.distributions.mixture_same_family.MixtureSameFamily,
    parameters_dict: dict[str, Any],
    em_gmm,
) -> None:
    """
    Plot Gaussian examtorchhyper.

    Args:
        args (Any): Arguments for plotting.
        trajectories (torch.Tensor): Trajectories of samples.
        train_loss (List[float]): Training loss values.
        val_loss (List[float]): Validation loss values.
        data_dist (torch.distributions.mixture_same_family.MixtureSameFamily):
            Data distribution.
        gmm_model (torch.distributions.mixture_same_family.MixtureSameFamily): GMM model.
        parameters_dict (dict[str, Any]): Dictionary of parameters.
    """

    # Plot the training and validation loss.
    plot_loss(
        args,
        train_loss['gmm'],
        val_loss['gmm'],
        log_scale=False,
    )
    if len(train_loss['ple']) > 1:
        plot_loss(
            args,
            train_loss['ple'],
            val_loss['ple'],
            log_scale=True,
            filename_ext='_ple',
            label_ext=' (ple penalty)',
        )

    for param, weight in zip(
            gmm_model.parameters(),
            parameters_dict.values(),
    ):
        param.data = weight.data

    if args.input_size == 2:

        # Generating the grid
        x_grid = torch.linspace(
            -10,
            10,
            500,
        ).to(data_dist.mixture_weights.device)
        y_grid = torch.linspace(
            -10,
            10,
            500,
        ).to(data_dist.mixture_weights.device)

        # Creating meshgrid
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')

        # Compute data log-likelihood.
        data_log_prob = -data_dist(torch.stack(
            [x_grid, y_grid],
            dim=-1,
        )).cpu()

        model_log_prob = -gmm_model(torch.stack(
            [x_grid, y_grid],
            dim=-1,
        )).cpu()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        # Plot the log-likelihoods.
        ax.contourf(
            x_grid.cpu(),
            y_grid.cpu(),
            data_log_prob,
            1000,
            vmin=-10,
        )
        # Remove grid lines
        ax.grid(False)
        # Change the size and font of tick labels on the X-axis
        ax.tick_params(axis='x', labelsize=12)

        # Set plot title and labels
        ax.set_title('Data log-likelihood')

        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 "trajectories_data.png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=500)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        ax.contourf(
            x_grid.cpu(),
            y_grid.cpu(),
            model_log_prob,
            1000,
            vmin=-10,
        )

        # Remove grid lines
        ax.grid(False)
        # Change the size and font of tick labels on the X-axis
        ax.tick_params(axis='x', labelsize=12)

        # Set plot title and labels
        ax.set_title('Model log-likelihood')

        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 "trajectories_model.png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=500)

    elif args.input_size == 1:
        x_grid = torch.linspace(
            -20,
            20,
            10000,
        ).to(data_dist.mixture_weights.device)

        data_log_prob = -data_dist(x_grid.view(-1, 1)).cpu()
        model_log_prob = -gmm_model(x_grid.view(-1, 1)).cpu()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()

        # Choose xlim based on values of data_log_prob.exp() that are larger
        # than 1e-4.
        x_min = x_grid[data_log_prob.exp() > 1e-6].min().item()
        x_max = x_grid[data_log_prob.exp() > 1e-6].max().item()
        ax.set_xlim([x_min, x_max])

        ax.plot(
            x_grid.cpu(),
            data_log_prob.exp(),
            label='Data log-likelihood',
            color='black',
            alpha=0.7,
            linestyle='--',
        )
        ax.plot(
            x_grid.cpu(),
            model_log_prob.exp(),
            label='Model log-likelihood',
            color='red',
            alpha=0.5,
        )

        # Add legend
        ax.legend()

        # Set plot title and labels
        ax.set_title('Data and model log-likelihood')

        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 "trajectories.png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=300)

    # Create a GMM object from the estimated parameters.
    estimated_gmm = GaussianMixtureModel(
        args.input_size,
        args.num_components,
    ).requires_grad_(False)
    estimated_gmm.custom_initialization(
        parameters_dict['mixture_weights'].cpu().numpy(),
        parameters_dict['mixture_means'].cpu().numpy(),
        parameters_dict['mixture_vars'].cpu().numpy(),
    )
    estimated_gmm = estimated_gmm.get_mixture_same_family_object()

    # Create a GMM object from the data distribution parameters.
    data_dist = data_dist.to('cpu')
    data_dist = data_dist.get_mixture_same_family_object()

    # Calculate the KL divergence between the estimated GMM and the data
    # GMM.
    kl_div = torch.distributions.kl.kl_divergence(data_dist, estimated_gmm)

    baseline_gmm = GaussianMixtureModel(
        args.input_size,
        args.num_components,
    ).requires_grad_(False)
    baseline_gmm.custom_initialization(
        em_gmm.weights_,
        em_gmm.means_,
        em_gmm.covariances_,
    )
    baseline_gmm = baseline_gmm.get_mixture_same_family_object()

    kl_div_baseline = torch.distributions.kl.kl_divergence(
        data_dist, baseline_gmm)

    # Write both KL divergences to a file.
    with open(os.path.join(plotsdir(args.experiment), 'kl_divergence.txt'),
              'w') as f:
        f.write(f'KL divergence between data and estimated GMM: {kl_div}\n')
        f.write(
            f'KL divergence between data and baseline GMM: {kl_div_baseline}\n'
        )
