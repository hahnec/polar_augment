import torch

# direct application
from rotation_mm import RandomMuellerRotation
rotate = RandomMuellerRotation(degrees=45, p=float('inf'))
mm_img = torch.randn([128, 128, 4, 4]).flatten(2, 3).permute(2, 0, 1)
mm_img_rotated = rotate(mm_img)

print(mm_img_rotated.shape)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(mm_img.mean(0))
plt.show()
plt.figure()
plt.imshow(mm_img_rotated.mean(0))
plt.show()

# calibration approach
from rotation_raw import RandomPolarRotation
rotate = RandomPolarRotation(degrees=45, p=float('inf'))
mm_img = torch.randn([128, 128, 4*3, 4]).flatten(2, 3).permute(2, 0, 1)
mm_img_rotated = rotate(mm_img)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(mm_img.mean(0))
plt.show()
plt.figure()
plt.imshow(mm_img_rotated.mean(0))
plt.show()