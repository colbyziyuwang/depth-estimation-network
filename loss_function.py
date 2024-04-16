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
        alpha = 0.5
        beta = 0.5
        gamma = 5.0
        

        super(MonocularDepthLoss, self).__init__()
    
    def reconstruct_image(image, disparity_map):
        IMAGE_SIZE = image.shape

        x_incidies -= disparity_map * (IMAGE_SIZE[1] * 0.1) # max disparity
        x_indicies_clamped = torch.clamp(x_incidies, 0, IMAGE_SIZE[1] - 1) # adjust to have it stay within range
        
        # interpolate pixel values
        x_indices_floor = torch.floor(x_indicies_clamped).type(torch.LongTensor)
        x_indices_ceil = torch.clamp(x_indices_floor + 1, 0, IMAGE_SIZE[1] - 1)

        interpolated_values_floor = torch.gather(image, 3, x_indices_floor)
        interpolated_values_ceil = torch.gather(image, 3, x_indices_ceil)

        interpolate_combine = (x_indices_ceil - x_indicies_clamped) * interpolated_values_floor + (x_indicies_clamped - x_indices_floor) * interpolated_values_ceil

        return interpolate_combine

    def forward(self, disparity_map_left, right_image, left_image, disparity_map_right=None):
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
        ssim_output = ssim.SSIM(window_size=5) * -1
        mse_function = nn.MSELoss()
        mse_ouput = mse_function(left_image, predicted_left_image)

        # smoothness term
        vertical_gradient = torch.square(left_image - torch.roll(left_image, 1, 2)).sum(1)
        horizontal_gradient = torch.square(left_image - torch.roll(left_image, 1, 3)).sum(1)

        vertical_disparity_gradient = torch.abs(disparity_map_left - torch.roll(disparity_map_left, 1, 2)).view(-1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        horizontal_disparity_gradient = torch.abs(disparity_map_left - torch.roll(disparity_map_left, 1, 3)).view(-1, IMAGE_SIZE[0], IMAGE_SIZE[1])

        smoothness_term = self.gamma * (torch.mean(vertical_disparity_gradient * torch.exp(-vertical_gradient)) + torch.mean(horizontal_disparity_gradient * torch.exp(-horizontal_gradient)))

        loss = self.alpha * ssim_output + self.beta * mse_ouput + smoothness_term
        return loss

