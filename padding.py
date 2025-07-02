import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def mirror_pad(frame, pad_h, pad_w):
    """MIrror around the image manually for circular padding."""
    C, H, W = frame.shape

    # Horizontal padding (left + right)
    left = torch.flip(frame[:, :, -pad_w:], dims=(2,))     # wrap from right
    right = torch.flip(frame[:, :, :pad_w], dims=(2,))     # wrap from left
    padded_w = torch.cat([left, frame, right], dim=2)

    # Vertical padding (top + bottom)
    top = torch.flip(padded_w[:, -pad_h:, :], dims=(1,))   # wrap from bottom
    bottom = torch.flip(padded_w[:, :pad_h, :], dims=(1,)) # wrap from top
    padded = torch.cat([top, padded_w, bottom], dim=1)

    return padded

def mirror_rotate(frame, angle, center=None, interpolation=False, expand=False):
    """
    Rotate an image tensor [C, H, W] using mirror-around padding.
    """
    C, H, W = frame.shape
    pad_h, pad_w = H, W  # pad one full image in each direction
    if center is None: center = [W/2, H/2]

    # Manually wrap-pad the image
    padded = mirror_pad(frame, pad_h, pad_w)

    # Rotate using torchvision
    rotated = TF.rotate(
        padded, angle,
        interpolation=InterpolationMode.NEAREST if not interpolation else interpolation,
        expand=expand,
        center=[center[0]+pad_w, center[1]+pad_h],
        fill=0  # ignored because border is mirror-padded
    )

    # Crop back to original size
    output = rotated[:, pad_h:pad_h+H, pad_w:pad_w+W]

    return output