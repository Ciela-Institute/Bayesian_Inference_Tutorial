import torch
import numpy as np
import physical_model as pm
import inference_tools

def grided_inference(noisy_image,noise_rms,centerX=0,centerY=0,extent=0.1):
  # Define the grid range and resolution
  x_min = centerX - extent/2
  x_max = centerX + extent/2
  y_min = centerY - extent/2
  y_max = centerY + extent/2

  # x_min, x_max = -1.0, 1.0
  # y_min, y_max = -1.0, 1.0
  grid_size = 100  # Number of points in each dimension

  # Generate meshgrid
  x_vals = torch.linspace(x_min, x_max, grid_size)
  y_vals = torch.linspace(y_min, y_max, grid_size)
  X_mesh, Y_mesh = torch.meshgrid(x_vals, y_vals, indexing='ij')


  # Evaluate log-posterior on the grid
  log_posterior_values = torch.zeros((grid_size, grid_size))

  for i in range(grid_size):
      for j in range(grid_size):
          log_posterior_values[i, j] = inference_tools.log_posterior(
              X_mesh[i, j], Y_mesh[i, j], noisy_image, noise_rms
          )
  log_posterior_values = (log_posterior_values - log_posterior_values.max())

  # Convert tensors to numpy for plotting
  X_mesh_np = X_mesh.numpy()
  Y_mesh_np = Y_mesh.numpy()

  # log_posterior_values is already a NumPy array
  log_posterior_values_np = log_posterior_values  
  return log_posterior_values_np , X_mesh_np , Y_mesh_np
