import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


def display_images(images):
  # Display images
  N = images.shape[0]
  fig, axes = plt.subplots(1, N, figsize=(15, 3))
  for i in range(N):
      axes[i].imshow(images[i].detach().cpu(), cmap='gray', extent=[-1, 1, -1, 1])
      axes[i].axis('off')
  plt.show()


def display_gridded_posterior(log_posterior_values_np, X_mesh_np, Y_mesh_np, ground_truth_x, ground_truth_y):
    """
    Displays a heatmap of the posterior distribution with contours and ground truth.

    Returns:
        fig, ax: Matplotlib figure and axis objects for later modifications.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure imshow aligns correctly with meshgrid by using the correct extent
    im = ax.imshow(
        log_posterior_values_np.T,  # Transpose to align correctly
        extent=[X_mesh_np.min(), X_mesh_np.max(), Y_mesh_np.min(), Y_mesh_np.max()], 
        origin='lower', 
        cmap='inferno', 
        aspect='equal'  # Ensure correct aspect ratio
    )

    # # Add contour lines using correctly aligned X_mesh and Y_mesh
    ax.contour(X_mesh_np, Y_mesh_np, log_posterior_values_np, colors='white', linewidths=0.5)

    # Mark the ground truth location correctly
    ax.scatter(
        ground_truth_x.item(), ground_truth_y.item(), 
        color='cyan', marker='x', s=100, label="Ground Truth"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # fig.colorbar(im, ax=ax)

    return fig, ax  # Return figure and axis objects





def plot_mcmc_sample(samples, ground_truth_x , ground_truth_y , ax=None, fig=None):

    # Create new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the MCMC samples
    ax.scatter(samples[:, :, 0], samples[:, :, 1], color="blue", s=1, alpha=0.3)
    ax.scatter(ground_truth_x, ground_truth_y,
               color="red", marker="x", s=1000, label="Ground Truth", linewidth=5)

    # Display updated figure
    display(fig)

    return fig, ax  # Return them for potential further modifications



def plot_langevin_sample(samples, ground_truth_x , ground_truth_y , ax=None, fig=None):

    # Create new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the MCMC samples
    ax.scatter(samples[:, 0], samples[:, 1], color="red", s=2, alpha=0.3)
    ax.scatter(ground_truth_x, ground_truth_y,
               color="red", marker="x", s=1000, label="Ground Truth", linewidth=5)

    # Display updated figure
    display(fig)

    return fig, ax  # Return them for potential further modifications




def plot_ABC_samples(samples, ax=None, fig=None):

    # Create new figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the MCMC samples
    ax.scatter(samples[:, 0], samples[:, 1], color="cyan", s=2, alpha=0.3)

    # Display updated figure
    display(fig)

    return fig, ax  # Return them for potential further modifications





