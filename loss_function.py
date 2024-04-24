import torch
import torch.nn as nn
import extlibs.pytorch_ssim as ssim
import torch.nn.functional as F

class MonocularDepthLoss(nn.modules.Module):
    def __init__(self, alpha=0.2, beta=1.0, gamma=1.0):
        super(MonocularDepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()
        
        x_og_coord = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_og_coord = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # shift horizontally
        x_shifts = disp[:, 0, :, :]  
        flow_field = torch.stack((x_og_coord + x_shifts, y_og_coord), dim=3)

        # grid sample coordinates shoudl be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def SSIM(self, x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x.pow(2)
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y.pow(2)
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
        SSIM = ssim_n / ssim_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)
    
    def get_gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def get_gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def get_disp_smoothness(self, disparity, image):
        disp_gradients_x = self.get_gradient_x(disparity)
        disp_gradients_y = self.get_gradient_y(disparity)

        image_gradients_x = self.get_gradient_x(image)
        image_gradients_y = self.get_gradient_y(image)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,
                     keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,
                     keepdim=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    def forward(self, disparity_map_left, left_image, right_image, disparity_map_right):
        # photo consistency
        predicted_left_img = self.apply_disparity(right_image, -disparity_map_left)
        predicted_right_img = self.apply_disparity(left_image, disparity_map_right) 

        l1_left = torch.mean(torch.abs(predicted_left_img - left_image)) 
        l1_right = torch.mean(torch.abs(predicted_right_img- right_image)) 
        ssim_left = torch.mean(self.SSIM(predicted_left_img,left_image)) # SSIM
        ssim_right = torch.mean(self.SSIM(predicted_right_img, right_image))

        image_loss_left = [self.alpha * ssim_left + (1 - self.alpha) * l1_left]
        image_loss_right = [self.alpha * ssim_right + (1 - self.alpha) * l1_right]
        image_loss = sum(image_loss_left + image_loss_right)

        # LR consistency
        RL_disparity = self.apply_disparity(disparity_map_right, -disparity_map_left) 
        LR_disparity = self.apply_disparity(disparity_map_left,  disparity_map_right) 
        LR_left_loss = [torch.mean(torch.abs(RL_disparity - disparity_map_left))] 
        LR_right_loss = [torch.mean(torch.abs(LR_disparity- disparity_map_right))]
        LR_loss = sum(LR_left_loss + LR_right_loss)

        # Disparities smoothness
        left_smoothness = self.get_disp_smoothness(disparity_map_left, left_image)
        right_smoothness = self.get_disp_smoothness(disparity_map_right, right_image)
        right_loss = [torch.mean(torch.abs(left_smoothness))]
        left_loss = [torch.mean(torch.abs(right_smoothness))]
        gradient_loss = sum(left_loss + right_loss)

        loss = image_loss + self.beta * gradient_loss + self.gamma * LR_loss

        return loss, predicted_left_img, predicted_right_img
