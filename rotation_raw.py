import math
import torch
import random
import numbers
import torchvision.transforms.functional as F


class RandomPolarRotation(object):
    """Rotate the raw polarimetry instrument data by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
        p (float): probability threshold with which image should be rotated or left untreated instead.
        any (bool): If false, only angles [90, 180, 270] will be chosen. Otherwise, a random angle 
            within the boundaries will be generated (default).

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0, p=0.5, any=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill
        self.p = p
        self.any = any

    def get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            params to be passed to ``rotate`` for random rotation.
        """
        
        angle = random.uniform(degrees[0], degrees[1]) if self.any else random.choice([90, 180, 270])

        return angle
    
    @staticmethod
    def get_rmat(deg):
        """Get rotation matrix for Mueller matrix.

        Returns:
            rotation matrix for Mueller matrix.
        """
        theta = deg / 180 * math.pi
        rmat = torch.tensor([
            [1, 0, 0, 0],
            [0, math.cos(2*theta), -math.sin(2*theta), 0],
            [0, math.sin(2*theta), +math.cos(2*theta), 0],
            [0, 0, 0, 1],
        ])

        return rmat

    def __call__(self, frame, label=None, transpose=True, angle=None, center=None, *args, **kwargs):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        if random.random() < self.p:
            # spatial transformation
            angle = self.get_params(self.degrees) if angle is None else angle
            self.center = self.center if center is None else center
            fill48 = torch.stack([torch.eye(4) for _ in range(3)]).flatten().tolist()
            frame = F.rotate(frame, angle, self.resample, self.expand, self.center, fill=fill48).moveaxis(0, -1)
            # unravel matrices
            I, A, W = frame[..., :16], frame[..., 16:32], frame[..., 32:]
            # HxWx16 to HxWx4x4 matrix reshaping
            shape = (*A.shape[:-1], 4, 4)
            zero_idcs = torch.all(A==torch.zeros_like(A), dim=-1)
            I, A, W = [el.reshape(shape) for el in [I, A, W]]
            if transpose: I, A, W = [el.transpose(-2, -1) for el in [I, A, W]]
            # replace zeros with identity matrices to make A and W invertible
            A[zero_idcs] = torch.eye(4, dtype=A.dtype, device=A.device)
            W[zero_idcs] = torch.eye(4, dtype=W.dtype, device=W.device)
            # mueller matrix transformation: A_theta = (R_theta @ A_inv)_inv since R_theta @ M @ R_-theta = R_theta @ A_inv @ I @ W_inv @ R_-theta
            T = self.get_rmat(angle).to(A.dtype)
            A = A @ torch.linalg.inv(T)
            W = T @ W
            # HxWx4 to HxWx16 matrix reshaping
            if transpose: I, A, W = [el.transpose(-2, -1) for el in [I, A, W]]
            I, A, W = [el.flatten(-2, -1).moveaxis(-1, 0) for el in [I, A, W]]
            # stack matrices together again
            rotated_frame = torch.cat([I, A, W], dim=0)
            if label is not None:
                rotated_label = F.rotate(label, angle, self.resample, self.expand, self.center, fill=self.fill)
                return rotated_frame, rotated_label
            return rotated_frame
        else:
            if label is not None:
                return frame, label
            return frame

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string