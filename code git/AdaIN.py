"""
AdaIN (Style Randomization module from "Reducing Domain Gap by Reducing Style Bias" paper

The input tensors (z and z') are feature maps from two different branches, each with shape (N, C, H, W) where:
N = batch size
C = number of channels (depth of the feature maps)
H and W = spatial height and width of the feature maps

The output (after applying AdaIN) will have the same shape as the input tensors (N, C, H, W).
"""

import torch
import torch.nn as nn

class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super(StyleRandomization, self).__init__()
        self.eps = eps 

    def forward(self, z: torch.Tensor, z_prime: torch.Tensor):
        """
        Apply style randomization (AdaIN) to convolutional feature maps from two branches.
        
        Arguments:
            z (Tensor): Feature map from the top branch with shape (N, C, H, W).
            z_prime (Tensor): Feature map from the bottom branch with shape (N, C, H, W).
        
        Returns:
            zt_style (Tensor): The output tensor after applying style randomization, with shape (N, C, H, W).
        """
        # Calculate mean and standard deviation for z (top branch)
        mu_z = z.mean(dim=(0, 2, 3), keepdim=True) # mean over (C, H, W)
        sigma_z = z.std(dim=(0, 2, 3), keepdim=True) + self.eps # std over (C, H, W)

        # Calculate mean and standard deviation for z' (bottom branch)
        mu_hat = z_prime.mean(dim=(0, 2, 3), keepdim=True) # mean over (C, H, W)
        sigma_hat = z_prime.std(dim=(0, 2, 3), keepdim=True) + self.eps # std over (C, H, W)

        # Apply AdaIN-style transformation (Eq.5 from the paper)
        zt_style = sigma_hat * ((z - mu_z) / sigma_z) + mu_hat

        return zt_style
