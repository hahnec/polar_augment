import torch

mm_img = torch.randn([128, 128, 4, 4]).flatten(2, 3).permute(2, 0, 1)
rw_img = torch.randn([128, 128, 4*3, 4]).flatten(2, 3).permute(2, 0, 1)

# direct application of rotation
from rotation_mm import RandomMuellerRotation
augment = RandomMuellerRotation(degrees=45, p=float('inf'))
mm_img_augment = augment(mm_img)
assert mm_img.shape == mm_img_augment.shape

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(mm_img.mean(0))
plt.show()
plt.figure()
plt.imshow(mm_img_augment.mean(0))
plt.show()

# direct application of flipping
from flip_mm import RandomMuellerFlip
augment = RandomMuellerFlip(orientation=1, p=float('inf'))
mm_img_augment = augment(mm_img)
assert mm_img.shape == mm_img_augment.shape

# calibration approach of rotation
from rotation_raw import RandomPolarRotation
augment = RandomPolarRotation(degrees=45, p=float('inf'))
rw_img_augment = augment(rw_img)
assert rw_img.shape == rw_img_augment.shape

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(rw_img.mean(0))
plt.show()
plt.figure()
plt.imshow(rw_img_augment.mean(0))
plt.show()

# calibration approach of flipping
from flip_raw import RandomPolarFlip
augment = RandomPolarFlip(orientation=2, p=float('inf'))
rw_img_augment = augment(rw_img)
assert rw_img.shape == rw_img_augment.shape
