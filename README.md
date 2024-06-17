# depth-estimation-network
Team members: Colby, Alex, Suyeong

# Descriptions
This project aims to estimate stereo disparities from pairs of left and right images using deep learning methods. The model is trained to generate disparity maps that can be used to reconstruct the depth information from the stereo images.

# Features
Disparity Map Calculation: Using a neural network to estimate disparity maps from stereo image pairs.

Image Reconstruction: Reconstructing depth information from the estimated disparities.

Loss Function Evaluation: Incorporating both Sum of Squared Differences (SSD) and Structural Similarity Index Measure (SSIM) for robust training.

Dataset Handling: Custom dataset class for efficient data loading and preprocessing.
