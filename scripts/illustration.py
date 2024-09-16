import numpy as np

r = np.arange(0, 180)

np.random.seed(3008*16)
n = np.random.rand(*r.shape)

from scipy.ndimage import gaussian_filter1d
n = gaussian_filter1d(n, sigma=1)

size = 250
half = size // 2
img = np.zeros([size, size])

for i in range(size):
    for j in range(size):
        y, x = (i-half), (j-half)
        angle_rad = np.arctan2(y, x)
        angle_deg = np.degrees(angle_rad)
        angle_deg = angle_deg % 180
        idx_c = int(np.round(angle_deg)) % 180
        idx_l = idx_c-1 if idx_c > 0 else 0
        idx_r = idx_c+1 if idx_c < 179 else 179
        wght_c = 1/abs(idx_c - angle_deg) if idx_c - angle_deg != 0 else 1
        wght_l = 1/abs(idx_l - angle_deg) if idx_c - angle_deg != 0 else 0
        wght_r = 1/abs(idx_r - angle_deg) if idx_c - angle_deg != 0 else 0
        wghts = np.array([wght_c, wght_l, wght_r]) / (wght_c + wght_l + wght_r)
        value = wghts[0] * n[idx_c] + wghts[1] * n[idx_l] + wghts[2] * n[idx_r]
        radius = (x**2 + y**2)**.5
        radius_norm = radius / (size**2 + size**2)**.5
        img[i, j] = value * np.cos(radius_norm*np.pi)**4

img = (img-img.min())/(img.max()-img.min())
img = np.round(255 * img).astype(np.uint8)

import imageio
imageio.imwrite('polar_intensities.png', img)

legend_opt = True

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.imshow(img, cmap='gray')
# horizontal
vx, vy = [0, size-1], [half, half]
ax.plot(vx, vy, color='green', label='horizontal')
# vertical
vx, vy = [half, half], [0, size-1]
ax.plot(vx, vy, color='blue', label='vertical')
# +45 degrees
vx, vy = [0.5, size-.5], [size-.5, 0.5]
ax.plot(vx, vy, color='orange', label='pos. 45 degrees')
# -45 degrees
vx, vy = [size-.5, 0.5], [size-.5, 0.5]
ax.plot(vx, vy, color='yellow', label='neg. 45 degrees')
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
if legend_opt: 
    plt.legend(
        loc=3, 
        title='Polarization directions',
        bbox_to_anchor=(1.05, 0.5), 
        fancybox = True,
        )
plt.tight_layout()
plt.savefig('polar_states.svg')
plt.show()