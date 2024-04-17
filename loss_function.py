"""
Monocular Depth Estimation Loss Function Module.

This module defines a custom loss function class used in monocular depth estimation tasks with neural networks. This is achieved via the MonocularDepthLoss class.

The class is structured to be filled in with specific loss calculation logic by Alex Wang. Note it is required to use PyTorch tensors to ensure compatibility with PyTorch's automatic differentiatio features, crucial for effective backpropagation in neural network training.

"""

import torch
import torch.nn as nn
import extlibs.pytorch_ssim as ssim

class MonocularDepthLoss(nn.Module):
    """
    This class defines a loss function used in monocular depth estimation.
    The loss function works as follows:
    1) Shift the pixels of the left image to generate the pseudo right image
    2) Shift the pixels of the right image to generate the pseudo left image
    3) Compare the pseudo right image and pseudo left image with real right image, and real
       left image, respectively. Use mean squared error for loss calculations
    """

    def __init__(self):
        """
        Initialize the object
        """
        super(MonocularDepthLoss, self).__init__()
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 5.0

    def reconstruct_image(self, image, disparity_map):
        IMAGE_SIZE = image.shape  # [N, C, H, W]

        # Calculate the shifted indices along the width (disparity adjustment)
        x_indices = disparity_map * (IMAGE_SIZE[3] * 0.1)  # Max disparity
        x_indices_clamped = torch.clamp(x_indices, 0, IMAGE_SIZE[3] - 1)

        # Floor and ceil indices for interpolation
        x_indices_floor = torch.floor(x_indices_clamped).long()
        x_indices_ceil = torch.clamp(x_indices_floor + 1, 0, IMAGE_SIZE[3] - 1)

        # Ensure indices are usable for gather; preparing for gather along width dimension
        # The gather operation should be applied across the width dimension
        gather_indices_floor = x_indices_floor.expand(-1, IMAGE_SIZE[1], -1, -1)  # Shape to [N, C, H, W]
        gather_indices_ceil = x_indices_ceil.expand(-1, IMAGE_SIZE[1], -1, -1)  # Shape to [N, C, H, W]

        # Gathering pixel values for floor and ceiling indices along the width dimension
        interpolated_values_floor = torch.gather(image, 3, gather_indices_floor)
        interpolated_values_ceil = torch.gather(image, 3, gather_indices_ceil)

        # Calculate weights for linear interpolation
        weights = (x_indices_clamped - x_indices_floor).expand_as(interpolated_values_floor)

        # Linear interpolation between floor and ceiling values
        interpolate_combine = weights * interpolated_values_ceil + (1 - weights) * interpolated_values_floor

        return interpolate_combine

    def forward(self, disparity_map_left, right_image, left_image, disparity_map_right):
        """
        Calculate the loss based on the input disparity map(s) and images.

        Parameters:
        disparity_map_left (Tensor): The predicted disparity map for the left image.
        right_image (Tensor): The right image tensor.
        left_image (Tensor): The left image tensor.
        disparity_map_right (Tensor, optional): The ground truth disparity map for the right image.

        Returns:
        Tensor: The calculated loss.
        """
        IMAGE_SIZE = left_image.shape

        # photoconsistency term
        predicted_left_image = self.reconstruct_image(right_image, disparity_map_left)
        ssim_object = ssim.SSIM(window_size=5)

        # Calculate SSIM using the forward method
        ssim_output = ssim_object.forward(left_image, predicted_left_image)
        mse_function = nn.MSELoss()
        mse_ouput = mse_function(left_image, predicted_left_image)

        # smoothness term
        # Calculate image gradients by squaring the differences after a roll operation
        vertical_gradient = torch.square(left_image - torch.roll(left_image, shifts=1, dims=2)).sum(1)  # Sum over the channel dimension
        horizontal_gradient = torch.square(left_image - torch.roll(left_image, shifts=1, dims=3)).sum(1)  # Sum over the channel dimension

        # Calculate disparity gradients
        vertical_disparity_gradient = torch.abs(disparity_map_left - torch.roll(disparity_map_left, shifts=1, dims=2))
        horizontal_disparity_gradient = torch.abs(disparity_map_left - torch.roll(disparity_map_left, shifts=1, dims=3))

        # Apply exponential decay on image gradients and multiply by disparity gradients
        vertical_term = torch.mean(vertical_disparity_gradient * torch.exp(-vertical_gradient))
        horizontal_term = torch.mean(horizontal_disparity_gradient * torch.exp(-horizontal_gradient))

        # Combine the smoothness terms weighted by gamma
        smoothness_term = self.gamma * (vertical_term + horizontal_term)

        loss = self.alpha * ssim_output + self.beta * mse_ouput + smoothness_term
        return loss

