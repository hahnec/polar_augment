import torch
import random
import numbers


class RandomMuellerFlip(object):
    """Flip the Mueller matrix frame by orientation.

    Args:
        orientation (int): Number indicating horizontal (0), vertical (1), or both directions (2).
        p (float): probability threshold (default: 0.5) with which image should be rotated or left untreated instead.
    """

    def __init__(self, orientation, p=0.5):
        if isinstance(orientation, numbers.Number):
            if orientation < 0:
                raise ValueError("Orientation must be a natural number.")
            self.orientation = orientation
        else:
            raise ValueError("Orientation must be a scalar.")
        self.p = float(p)

    def get_fmat(self):
        """Get flip matrix for Mueller matrix.

        Returns:
            flip matrix for Mueller matrix.
        """

        f = -1 if self.orientation in [0, 1] else 1

        rmat = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, f, 0],
            [0, 0, 0, f],
        ])

        return rmat
    
    def flip_img(self, img):

        dims = [-1, -2] if self.orientation == 2 else [-self.orientation-1,]

        return torch.flip(img, dims=dims)

    def __call__(self, frame, label=None, transpose=True, *args, **kwargs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Flipped image.
        """

        if random.random() < self.p:
            # spatial transformation
            frame = self.flip_img(frame).moveaxis(0, -1)
            # HxWx16 to HxWx4x4 matrix reshaping
            shape = (*frame.shape[:-1], 4, 4)
            frame = frame.reshape(shape)
            if transpose: frame = frame.transpose(-2, -1)
            # mueller matrix transformation: A_theta = (R_theta @ A_inv)_inv since R_theta @ M @ R_-theta = R_theta @ A_inv @ I @ W_inv @ R_-theta
            T = self.get_fmat().to(frame.dtype)
            frame = T @ frame @ T.transpose(-2, -1)
            # HxWx4 to HxWx16 matrix reshaping
            if transpose: frame = frame.transpose(-2, -1)
            flipped_frame = frame.flatten(-2, -1).moveaxis(-1, 0)
            # stack matrices together again
            if label is not None:
                flipped_label = self.flip_img(label)
                return flipped_frame, flipped_label
            return flipped_frame
        else:
            if label is not None:
                return frame, label
            return frame

