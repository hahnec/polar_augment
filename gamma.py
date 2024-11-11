import cv2
import numpy as np

class GammaAugmentation:
    def __init__(self, gamma_range=(0.5, 2.0)):
        """
        Initialize the gamma augmentation with a specified range.
        
        Parameters:
        gamma_range (tuple): Tuple specifying the range of gamma values to apply. 
                             gamma < 1 darkens the image, gamma > 1 brightens the image.
        """
        self.gamma_range = gamma_range

    def apply(self, image, label=None):
        """
        Apply gamma augmentation to the given image.
        
        Parameters:
        image (numpy.ndarray): Input image to augment.
        
        Returns:
        (numpy.ndarray, Any): Gamma-corrected image, label
        """
        # Randomly choose a gamma value within the specified range
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        image = cv2.LUT(image, table)

        return image, label

    def __call__(self, image, label=None):
        """
        Callable interface for applying the gamma augmentation.
        
        Parameters:
        image (numpy.ndarray): Input image to augment.
        
        Returns:
        numpy.ndarray: Gamma-corrected image.
        """
        return self.apply(image, label)
