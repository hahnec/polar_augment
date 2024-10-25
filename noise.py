import torch
import random


class RandomGaussNoise(object):
    """Add Gaussion noise to image data.

    Args:
        mean: mean or noise center of the distribution (default 0).
        std: standard deviation or spread of the distribution (default 0).
        p (float): probability threshold with which image should be rotated or left untreated instead.
        sample_std: If True, sample the spread of the Gaussian distribution uniformly from 0 to std.
    """

    def __init__(self, mean=0.0, std=0.1, p=0.5, sample_std=False):
        if std < 0:
            raise ValueError("std must be positive.")
        self.mean = mean
        self.std = std
        self.p = p
        self.sample_std = sample_std

    def __call__(self, frame, label=None, transpose=True, mean=None, *args, **kwargs):
        """
        Args:
            img (PIL Image): Image to be disturbed.

        Returns:
            PIL Image: Noisy image.
        """

        if random.random() < self.p:
            # create noisy frame
            std = torch.rand(1).item() * self.std if self.sample_std else self.std
            mean = self.mean if mean is None else mean
            noise = torch.normal(mean, std, size=frame.shape)
            frame = frame + noise
            if label is not None:
                # leave label untreated
                return frame, label
            return frame
        else:
            if label is not None:
                return frame, label
            return frame
