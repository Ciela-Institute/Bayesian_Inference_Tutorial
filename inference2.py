import numpy as np
import emcee
import inference_tools
import torch

def MCMC_inference(data,noise_rms):
    # Define the log-probability function for emcee
    def log_prob(params, data, noise_rms):
        x, y = params
        return inference_tools.log_posterior(torch.tensor(x), torch.tensor(y), data, noise_rms).item()  # Convert to scalar

    # Number of walkers and dimensions
    num_walkers = 64  # Number of MCMC chains
    ndim = 2  # We are sampling (x, y)

    # Start walkers near the posterior peak with a small Gaussian spread
    initial_x = np.random.normal(loc=0, scale=0.1, size=num_walkers)
    initial_y = np.random.normal(loc=0, scale=0.1, size=num_walkers)
    initial_positions = np.vstack([initial_x, initial_y]).T  # Shape (num_walkers, ndim)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_prob, args=[data, noise_rms])

    # Run MCMC for 1000 steps
    num_steps = 1000
    sampler.run_mcmc(initial_positions, num_steps, progress=True)


    samples = sampler.get_chain(discard=500,flat=False)  # Shape (num_samples, ndim)

    return samples