"""
Unet.py: U-Net Architecture for Disparity Estimation

This module implements the U-Net architecture, a convolutional neural network optimized for tasks requiring precise localization, such as disparity estimation from stereo images. The U-Net model is characterized by its symmetric structure, featuring a contracting path to capture context and an expanding path that enables precise localization. This architecture is particularly suited for image-to-image tasks where the output is a dense map, such as disparity estimation in stereo vision.

The U-Net implemented in this module is adapted for disparity estimation, with modifications to suit the specific requirements of the task. It includes layers for feature extraction, downsampling, upsampling, and concatenation of feature maps to preserve spatial information crucial for accurate disparity calculation.

Functions:
- __init__(): Initializes a new instance of the U-Net model, constructing the network architecture with the specified parameters and preparing it for disparity estimation.
- forward(): Defines the forward pass of the U-Net model, taking an input image or image pair and producing the corresponding disparity map as output.

Usage:
This module can be used to instantiate a U-Net model for disparity estimation, allowing for training the model on a dataset of stereo image pairs and using the trained model for predicting disparity maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    UNet Architecture for Stereo Image Disparity Estimation.

    This implementation of the U-Net architecture is specifically tailored for processing images from the Tsukuba dataset, which are of size 288x384 pixels. The U-Net model is a convolutional neural network that excels at tasks requiring precise localization, such as disparity estimation in stereo vision. It features a symmetric encoder-decoder structure with skip connections, enabling the capture of context at various resolutions and the precise localization of features.

    The model is designed to accept input images of size 288x384, aligning with the dimensions of the Tsukuba stereo image pairs. It applies a series of convolutional, activation, and pooling layers to downsample the input, then progressively upsamples and concatenates the feature maps to predict the disparity map with the same resolution as the input.

    Methods:
        __init__: Initializes the U-Net model with the necessary layers and parameters tailored for the Tsukuba dataset's image size.
        forward: Perform the forward step for input image x
    """
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        # First encoder block
        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1)
        self.enc_relu1 = nn.ReLU()
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # First pooling layer reduces size by half

        # Second encoder block
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        self.enc_relu2 = nn.ReLU()
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Second pooling layer

        # Third encoder block
        self.enc_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1)
        self.enc_relu3 = nn.ReLU()
        self.enc_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Third pooling layer

        # Fourth encoder block
        self.enc_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1)
        self.enc_relu4 = nn.ReLU()
        self.enc_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Fourth pooling layer reduces size to its minimum

        # Decoder
        # First decoder block (upsampling + concatenation + convolution)
        self.dec_upsample1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2) # first upsample
        self.dec_conv1 = nn.Conv2d(in_channels=256 + 256, out_channels=128, kernel_size=(3,3), padding=1)  # Convolution after concatenation with enc_conv3's output

        # Second decoder block
        self.dec_upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)  # Second upsample
        self.dec_conv2 = nn.Conv2d(in_channels=64 + 128, out_channels=64, kernel_size=(3,3), padding=1)  # Convolution after concatenation with enc_conv2's output

        # Third decoder block
        self.dec_upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)  # Third upsample
        self.dec_conv3 = nn.Conv2d(in_channels=32 + 64, out_channels=32, kernel_size=(3,3), padding=1)  # Convolution after concatenation with enc_conv1's output

        # Fourth (final) decoder block
        self.dec_upsample4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)  # Last upsample
        self.dec_conv4a = nn.Conv2d(in_channels=16 + 3, out_channels=1, kernel_size=(3,3), padding=1)  # Final convolution combines features with original image
        self.dec_conv4b = nn.Conv2d(in_channels=16 + 3, out_channels=1, kernel_size=(3,3), padding=1)

    def forward(self, x):
        """
        Forward pass of the UNet model.

        This function defines how the input tensor 'x' flows through the network. The input passes sequentially through multiple encoding layers, followed by corresponding decoding layers with skip connections from the encoder. Each encoding layer consists of a convolution followed by a ReLU activation and a max pooling operation. In the decoder, the feature maps are upsampled, concatenated with the corresponding encoder feature maps, and then passed through a convolutional layer.

        The forward pass ensures that the spatial dimensions of the output match that of the input, making it suitable for pixel-wise prediction tasks such as disparity estimation.

        Parameters:
            x (torch.Tensor): The input tensor with shape (N, C, H, W), where
                N is the batch size,
                C is the number of channels,
                H is the height,
                W is the width.

        Returns:
            torch.Tensor: The output tensor with predicted values, having the same height and width as the input but with a single channel (N, 1, H, W).
        """
        # batchify x
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Adds a dimension of size 1 at the specified position 0

        # Encoder
        enc1 = self.enc_relu1(self.enc_conv1(x))
        pool1 = self.enc_pool1(enc1)

        enc2 = self.enc_relu2(self.enc_conv2(pool1))
        pool2 = self.enc_pool2(enc2)

        enc3 = self.enc_relu3(self.enc_conv3(pool2))
        pool3 = self.enc_pool3(enc3)

        enc4 = self.enc_relu4(self.enc_conv4(pool3))
        pool4 = self.enc_pool4(enc4)

        # Decoder     
        # Upsample, then concatenate with the corresponding encoder output
        up1 = self.dec_upsample1(pool4)

        # Concatenate along the channel dimension
        dec1 = torch.cat((up1, pool3), dim=1)
        dec1 = self.dec_conv1(dec1)

        up2 = self.dec_upsample2(dec1)
        dec2 = torch.cat((up2, pool2), dim=1)
        dec2 = self.dec_conv2(dec2)

        up3 = self.dec_upsample3(dec2)
        dec3 = torch.cat((up3, pool1), dim=1)
        dec3 = self.dec_conv3(dec3)

        up4 = self.dec_upsample4(dec3)
        
        # The initial image 'x' is concatenated in the last layer
        dec4 = torch.cat((up4, x), dim=1)
        dec4a = self.dec_conv4a(dec4)
        dec4b = self.dec_conv4b(dec4)

        # disparity map
        right_disparity = dec4a
        left_disparity = dec4b
         
        # if x's batch dim is 1 then remove it
        if right_disparity.size(0) == 1:
            right_disparity = right_disparity.squeeze(0).squeeze(0)
            left_disparity = left_disparity.squeeze(0).squeeze(0)

        # right disparity is positive
        return right_disparity, left_disparity

if __name__ == "__main__":
    # Initialize your network
    net = UNet()

    # Create a dummy input tensor that matches the expected input dimensions
    dummy_input = torch.randn(1, 3, 288, 384)  # (batch, channel, height, width)

    # Perform a forward pass
    try:
        output_right, output_left = net(dummy_input)
        print("Forward pass successful. Output shape:", output_left.shape)
    except Exception as e:
        print("Error during forward pass:", e)
