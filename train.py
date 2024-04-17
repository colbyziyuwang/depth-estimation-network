# train the network

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from Unet import UNet
from loss_function import MonocularDepthLoss

# Initialize the U-Net model and loss function
model = UNet()
loss_function = MonocularDepthLoss()

# Create a dummy left image and right image (single image, HWC format)
image_l = torch.rand(1, 3, 256, 256)  # 256x256 size, 3 channels
image_r = torch.rand(1, 3, 256, 256)  # same as above

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs to train (set high to see if it can overfit)
epochs = 100

# List to record losses
loss_history = []

# Training loop
model.train()  # Set model to training mode
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    optimizer.zero_grad()

    # Forward pass
    right_disparity, left_disparity = model(image_l)

    # Calculate loss
    loss = loss_function(left_disparity, image_r, image_l, right_disparity)
    loss_history.append(loss.item())

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

print("Training complete. The model should have overfit to the single image.")

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot as a PNG file
plt.savefig('training_loss.png')  # You can specify a different path or file name

# Close the plot
plt.close()
