import torch
import matplotlib.pyplot as plt

noise_rms = 0.4


def population_distribution_covariance_matrix(var_x=0.02, var_y=0.02, corr_xy=0.5):
    """
    Construct a covariance matrix given variances and correlation coefficients.
    
    Parameters:
        var_x, var_y (float): Variances of x and y.
        corr_xy (float): Correlation coefficient between x and y.
        
    Returns:
        cov_matrix (torch.Tensor): 2x2 covariance matrix.
    """
    std_x = torch.sqrt(torch.tensor(var_x))
    std_y = torch.sqrt(torch.tensor(var_y))
    
    cov_matrix = torch.tensor([
        [var_x, corr_xy * std_x * std_y],
        [corr_xy * std_x * std_y, var_y]
    ])
    
    return cov_matrix

def sample_from_population_distribution(N):
    mean = torch.tensor([0.0, 0.0])
    cov = population_distribution_covariance_matrix()
    
    """
    Sample N sets of circle parameters (x, y) from a multivariate normal distribution.
    
    Parameters:
        N (int): Number of samples to generate.
        
    Returns:
        X (torch.Tensor): Sampled parameters of shape (N, 2).
    """
    return torch.distributions.MultivariateNormal(mean, cov).sample((N,))

def generate_image_from_parameter(X, image_size=64, fixed_r=0.4, edge_width=0.2):
    """
    Generate differentiable images of soft-edged circles based on parameters.
    """
    N = X.shape[0]
    images = torch.zeros((N, image_size, image_size), dtype=torch.float32)

    linspace = torch.linspace(-1, 1, image_size, dtype=torch.float32)
    X_grid, Y_grid = torch.meshgrid(linspace, linspace, indexing='ij')

    for i in range(N):
        x, y = X[i]
        distance = torch.sqrt((X_grid - x) ** 2 + (Y_grid - y) ** 2)

        # Use a sigmoid for differentiable transition instead of clamp
        mask = torch.sigmoid(-(distance - fixed_r) / edge_width)

        images[i] = mask

    return images
    
def generate_noise(N, rms=0.1):
    """
    Generate a realization of Gaussian noise with a given RMS level.
    
    Parameters:
        N (int): Number of samples.
        rms (float): Root mean square of the noise level.
        
    Returns:
        noise (torch.Tensor): Tensor of shape (N, 64, 64) containing Gaussian noise.
    """
    image_size = 64
    return torch.normal(mean=0.0, std=rms, size=(N, image_size, image_size))

def generate_noisy_data(N):
    """
    Generate noisy observations of circles.
    
    Parameters:
        N (int): Number of samples.
        
    Returns:
        noisy_images (torch.Tensor): Tensor of shape (N, 64, 64) containing noisy images.
        noise_rms (float): RMS of the noise.
        ground_truth_parameters (torch.Tensor): Ground truth circle parameters of shape (N, 2).
    """
    ground_truth_parameters = sample_from_population_distribution(N)
    ground_truth_images = generate_image_from_parameter(ground_truth_parameters)
    noise = generate_noise(N, rms=noise_rms)
    noisy_images = ground_truth_images + noise
    return noisy_images, noise_rms, ground_truth_parameters
