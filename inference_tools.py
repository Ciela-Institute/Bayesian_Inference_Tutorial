import torch
import physical_model as pm
prior_rms = 0.3

def log_gaussian_likelihood(noisy_image, model_image, noise_rms):
    """
    Compute the log Gaussian likelihood density for a given noisy image and model image.
    
    Parameters:
        noisy_image (torch.Tensor): Observed noisy image of shape (64, 64).
        model_image (torch.Tensor): Model image of shape (64, 64).
        noise_rms (float): Standard deviation of the Gaussian noise.
        
    Returns:
        log_likelihood (torch.Tensor): Scalar tensor representing the log-likelihood density.
    """
    residual = noisy_image - model_image
    log_likelihood = -0.5 * torch.sum((residual / noise_rms) ** 2)
    return log_likelihood


def log_gaussian_prior(x, y):
    """
    Compute the log Gaussian prior for the parameters x and y.
    
    Parameters:
        x (float or torch.Tensor): X-coordinate of the circle center.
        y (float or torch.Tensor): Y-coordinate of the circle center.
        prior_rms (float): Standard deviation of the Gaussian prior (default: 100).
        
    Returns:
        log_prior (torch.Tensor): Scalar tensor representing the log-prior density.
    """
    prior_variance = prior_rms ** 2
    log_prior = -0.5 * ((x ** 2 + y ** 2) / prior_variance)
    return log_prior

def sample_prior():
    x = torch.normal(0.0, prior_rms, size=(1,)).item()
    y = torch.normal(0.0, prior_rms, size=(1,)).item()
    params = torch.tensor([[x, y]])
    return params



def log_posterior(x, y, noisy_image, noise_rms):
    """
    Compute the log posterior by summing the log likelihood and log prior.

    Parameters:
        x (torch.Tensor): X-coordinate of the circle center (must have requires_grad=True).
        y (torch.Tensor): Y-coordinate of the circle center (must have requires_grad=True).
        noisy_image (torch.Tensor): Observed noisy image of shape (64, 64).
        noise_rms (float): Standard deviation of the Gaussian noise.

    Returns:
        log_posterior (torch.Tensor): Scalar tensor representing the log-posterior density.
    """
    # Ensure x and y are already PyTorch tensors with gradients enabled
    x.requires_grad_(True)  # Ensure autograd tracks x
    y.requires_grad_(True)  # Ensure autograd tracks y

    # Generate model image while preserving gradients
    model_image = pm.generate_image_from_parameter(torch.stack([x, y]).view(1, 2))[0]  # Use torch.stack to maintain autograd

    # Compute log-likelihood
    log_likelihood_value = log_gaussian_likelihood(noisy_image, model_image, noise_rms)

    # Compute log-prior
    log_prior_value = log_gaussian_prior(x, y)

    return log_likelihood_value + log_prior_value
    