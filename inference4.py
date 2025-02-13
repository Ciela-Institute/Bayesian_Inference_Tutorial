import torch
import numpy as np
import physical_model as pm
import inference_tools  # Ensure this is imported
import matplotlib.pyplot as plt

def abc_inference(data, noise_rms, epsilon=0.1, num_trials=10000):
    """
    Perform Approximate Bayesian Computation (ABC) to infer (x, y) parameters.

    Parameters:
        data (torch.Tensor): Observed noisy image (64x64).
        noise_rms (float): RMS of Gaussian noise.
        epsilon (float): Tolerance for accepting samples.
        num_trials (int): Number of prior samples to generate.

    Returns:
        accepted_samples (torch.Tensor): Accepted (x, y) samples of shape (N, 2).
    """
    accepted_samples = []

    for _ in range(num_trials):
        # Sample from prior
        params = inference_tools.sample_prior()  # Shape: (1, 2)

        # Generate simulated image
        simulated_image = pm.generate_image_from_parameter(params)[0]  # Shape: (64, 64)

        # Add Gaussian noise
        noise = pm.generate_noise(1, rms=noise_rms)[0]  # Shape: (64, 64)
        simulated_noisy_image = simulated_image + noise

        # Compute absolute difference
        difference = torch.abs(simulated_noisy_image - data)

        # Check if every pixel difference is within epsilon
        if torch.all(difference < epsilon):
            accepted_samples.append(params[0].clone().detach())  # Ensure tensor is not linked to computation graph

    # Convert list of tensors to a single tensor
    if len(accepted_samples) == 0:
        print("Warning: No samples accepted! Try increasing epsilon or num_trials.")
        return torch.empty((0, 2))  # Return empty (N,2) tensor

    return torch.stack(accepted_samples).detach().cpu().numpy()  # Stack tensors into shape (N, 2)


