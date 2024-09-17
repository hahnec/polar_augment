import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

plt.rc('text', usetex=False)
plt.rc('text.latex')

if __name__ == '__main__':

    size = 251
    lw = 8
    a = 0.8
    filter_opt = False
    legend_opt = True
    axes_opt = False

    r = np.arange(0, 180)

    np.random.seed(3008*16)
    n = np.random.rand(*r.shape)

    if filter_opt:
        n = scipy.ndimage.gaussian_filter1d(n, sigma=1)
    else:
        n = np.sin(r/30*np.pi) #+ n
        n = n - n.min()

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

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    if axes_opt:
        # horizontal
        vx, vy = [0, size-1], [half, half]
        ax.plot(vx, vy, lw=lw, color='green', alpha=a, label='horizontal')
        # vertical
        vx, vy = [half, half], [0, size-1]
        ax.plot(vx, vy, lw=lw, color='blue', alpha=a, label='vertical')
        # +45 degrees
        vx, vy = [0.5, size-.5], [size-.5, 0.5]
        ax.plot(vx, vy, lw=lw, color='gray', alpha=a, label='pos. 45 degrees')
        # -45 degrees
        vx, vy = [size-.5, 0.5], [size-.5, 0.5]
        ax.plot(vx, vy, lw=lw, color='yellow', alpha=a, label='neg. 45 degrees')
    else:
        # Set x and y axis limits to match image dimensions
        height, width = img.shape
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # invert y-axis for image coordinates

        # Add arrows for axes using annotate
        ax.annotate('', xy=(width, height//2), xytext=(0, height//2),
                    arrowprops=dict(facecolor='white', edgecolor='white', shrink=0.05, width=4, headwidth=32, headlength=40))
        ax.annotate('', xy=(width//2, 0), xytext=(width//2, height), 
                    arrowprops=dict(facecolor='white', edgecolor='white', shrink=0.05, width=4, headwidth=32, headlength=40))
        
        # draw labels
        gap = 30
        ax.annotate(r'$p_x$', xy=(width-gap, height//2+gap), color='white', fontsize=86)
        ax.annotate(r'$p_y$', xy=(width//2-gap, gap), color='white', fontsize=86)

        # Optional: Add grid lines (example at intervals of 50 pixels)
        x_ticks = np.arange(0, width, 50)
        y_ticks = np.arange(0, height, 50)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, color='white', linestyle=':', linewidth=0.5)
    if axes_opt and legend_opt: 
        plt.legend(
            loc=3, 
            title='Polarization directions',
            bbox_to_anchor=(1.05, 0.5), 
            fancybox = True,
            )
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    
    identity = lambda arr: arr
    rotate = lambda arr: scipy.ndimage.rotate(arr, angle=30//2, reshape=False)
    vflip = lambda arr: arr[::-1, :]
    hflip = lambda arr: arr[:, ::-1]
    fflip = lambda arr: hflip(vflip(arr))
    for i, fun in enumerate([identity, rotate, vflip, hflip, fflip]):
        ax.imshow(fun(img), cmap='gist_heat')
        plt.tight_layout()
        plt.savefig('./docs/polar_states_'+str(i)+'.svg')
    plt.show()