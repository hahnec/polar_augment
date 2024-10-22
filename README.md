# polar_augment

## Description

This repository provides polarimetric image augmentations such as an [SO(2) rotation](simulate_rotation_script.py).

## Example showing polarimetric azimuth maps after SO(2) rotation

| ![RotationAnimation](docs/animation_with_alpha_wo.gif) | ![RotationAnimation](docs/animation_with_alpha_rect.gif) | ![GT](docs/gt.png) |
|:--------------------------:|:--------------------------:|:--------------------------:|
| **spatial-only rotation** | **spatial + polarimetric rotation** | **ground truth** |

<br>
<p align="center">
  <img src="docs/color_bar.svg" alt="Colorbar" width="33%" />
</p>

## Installation

```bash
$ git clone github.com/hahnec/polar_augment
$ cd polar_augment
$ bash install.sh
```

### Usage

The provided transforms expect the image dimensions to be in PyTorch style `CxHxW`.

```python
import torch

# direct application
from polar_augment.rotation_mm import RandomMuellerRotation
rotate = RandomMuellerRotation(degrees=45, p=float('inf'))
mm_img = torch.randn([128, 128, 4, 4]).flatten(2, 3).permute(2, 0, 1)
mm_img_rotated = rotate(mm_img)
print(mm_img_rotated.shape)

# application for calibration matrices (dataloader-friendly for raw data)
from polar_augment.rotation_raw import RandomPolarRotation
rotate = RandomPolarRotation(degrees=45, p=float('inf'))
mm_img = torch.randn([128, 128, 4*3, 4]).flatten(2, 3).permute(2, 0, 1)
mm_img_rotated = rotate(mm_img)
print(mm_img_rotated.shape)

```

Alternatively, the transforms can be integrated during dataloading as for example by

```python
from torchvision.transforms import ToTensor
from polar_augment.flip_raw import RandomPolarFlip
from polar_augment.rotation_raw import RandomPolarRotation

# define list of transforms
transforms = [
        ToTensor(), 
        RandomPolarRotation(degrees=180, p=.5), # rotation
        RandomPolarFlip(orientation=0, p=.5),   # horizontal flip
        RandomPolarFlip(orientation=1, p=.5),   # vertical flip
        RandomPolarFlip(orientation=2, p=.5),   # combined horizontal and vertical flip
    ]

# pass transforms to your dataset
PolarimetryDataset(some_file_path, transforms=transforms)

```

where the augmentation can then be applied to frames and labels within the dataset

```python

for transform in self.transforms:
    frames, labels = transform(frames, label=labels)

```