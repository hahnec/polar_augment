import torch

class GammaAugmentation:
    def __init__(self, gamma_range=(0.5, 2.0)):
        """
        Initialize the gamma augmentation with a specified range.
        
        Parameters:
        gamma_range (tuple): Tuple specifying the range of gamma values to apply. 
                             gamma < 1 darkens the image, gamma > 1 brightens the image.
        """
        self.gamma_range = gamma_range

    def apply(self, image):
        """
        Apply gamma augmentation to the given image.
        
        Parameters:
        image (torch.Tensor): Input image tensor to augment, with pixel values in [0, 1].
        
        Returns:
        torch.Tensor: Gamma-corrected image tensor.
        """
        # Randomly choose a gamma value within the specified range
        gamma = torch.empty(1).uniform_(self.gamma_range[0], self.gamma_range[1]).item()
        
        # Apply symmetric gamma correction: image' = image ^ gamma
        image = torch.sign(image) * torch.abs(image).clamp(min=1e-8).pow(gamma)
        
        return image

    def __call__(self, image, label=None):
        """
        Callable interface for applying the gamma augmentation.
        
        Parameters:
        image (torch.Tensor): Input image tensor to augment, with pixel values in [0, 1].
        
        Returns:
        torch.Tensor: Gamma-corrected image tensor.
        """
        return self.apply(image), label
