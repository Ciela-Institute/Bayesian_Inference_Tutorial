import torch
import inference_tools
import numpy as np

def langevin_sampler(log_posterior_func, initial_params, step_size=0.01, num_steps=1000):
    """
    Performs Unadjusted Langevin Sampling (ULA) using the gradient of the log posterior.
    """
    theta = initial_params.clone().detach().requires_grad_(True)
    samples = []

    for _ in range(num_steps):
        theta.requires_grad_(True)  # Ensure gradient tracking at each step

        log_posterior = log_posterior_func(theta[0], theta[1])  # Compute log posterior
        
        log_posterior.backward()

        if theta.grad is None:
            raise RuntimeError("Gradient computation failed. Check log_posterior_func.")

        # Langevin update step
        with torch.no_grad():
            gradient = theta.grad  # Extract gradient
            brownian_noise = torch.randn_like(theta)  # Gaussian noise
            theta += (step_size / 2) * gradient + torch.sqrt(torch.tensor(step_size)) * brownian_noise

            theta.grad.zero_()  # Reset gradients

            samples.append(theta.clone().detach())  # Store sample

    return torch.stack(samples)


def Langevin_Inference(N,data,noise_rms):
  # Define initial starting point
  initial_params = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True)

  # Run Langevin sampling
  samples = langevin_sampler(
      lambda x, y: inference_tools.log_posterior(x, y, data, noise_rms),
      initial_params,
      step_size=0.0001,
      num_steps=N
  )

  # Convert samples to numpy for plotting
  x_samples, y_samples = samples[:, 0].numpy(), samples[:, 1].numpy()
  langevin_samples = np.column_stack((x_samples, y_samples))
  return langevin_samples
