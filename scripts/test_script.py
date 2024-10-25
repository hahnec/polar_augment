import time
import torch

mm_img = torch.randn([388, 516, 4, 4]).flatten(2, 3).permute(2, 0, 1)
rw_img = torch.randn([388, 516, 4*3, 4]).flatten(2, 3).permute(2, 0, 1)

times = []
iter_num = 50
plot_opt = False

# direct application of rotation
from rotation_mm import RandomMuellerRotation
augment = RandomMuellerRotation(degrees=45, p=float('inf'))
for i in range(iter_num):
    start = time.perf_counter()
    mm_img_augment = augment(mm_img)
    times.append(time.perf_counter()-start)
assert mm_img.shape == mm_img_augment.shape

print(sum(times)/len(times))
times = []

if plot_opt:
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
for i in range(iter_num):
    start = time.perf_counter()
    mm_img_augment = augment(mm_img)
    times.append(time.perf_counter()-start)
assert mm_img.shape == mm_img_augment.shape

print(sum(times)/len(times))
times = []

# calibration approach of rotation
from rotation_raw import RandomPolarRotation
augment = RandomPolarRotation(degrees=45, p=float('inf'))
for i in range(iter_num):
    start = time.perf_counter()
    rw_img_augment = augment(rw_img)
    times.append(time.perf_counter()-start)
assert rw_img.shape == rw_img_augment.shape

print(sum(times)/len(times))
times = []

if plot_opt:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(rw_img.mean(0))
    plt.show()
    plt.figure()
    plt.imshow(rw_img_augment.mean(0))
    plt.show()

# calibration approach of flipping
from flip_raw import RandomPolarFlip
augment = RandomPolarFlip(orientation=1, p=float('inf'))
for i in range(iter_num):
    start = time.perf_counter()
    rw_img_augment = augment(rw_img)
    times.append(time.perf_counter()-start)
assert rw_img.shape == rw_img_augment.shape

print(sum(times)/len(times))
times = []

import torchvision.transforms.functional as TF

for i in range(iter_num):
    start = time.perf_counter()
    angle = torch.empty(1).uniform_(-45, 45).item()  # Random angle between -45 and 45 degrees
    augmented_img = TF.rotate(mm_img, angle=angle)
    times.append(time.perf_counter()-start)
assert mm_img.shape == augmented_img.shape

print(sum(times)/len(times))
times = []

for i in range(iter_num):
    start = time.perf_counter()
    angle = torch.empty(1).uniform_(-45, 45).item()  # Random angle between -45 and 45 degrees
    augmented_img = TF.rotate(rw_img, angle=angle)
    times.append(time.perf_counter()-start)
assert rw_img.shape == augmented_img.shape

print(sum(times)/len(times))
times = []

for i in range(iter_num):
    start = time.perf_counter()
    augmented_img = TF.vflip(mm_img)
    times.append(time.perf_counter()-start)
assert mm_img.shape == augmented_img.shape

print(sum(times)/len(times))
times = []

for i in range(iter_num):
    start = time.perf_counter()
    augmented_img = TF.vflip(rw_img)
    times.append(time.perf_counter()-start)
assert rw_img.shape == augmented_img.shape

print(sum(times)/len(times))
times = []