import torch
import torchvision
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from mm.models import MuellerMatrixSelector
from mm.utils.cod import read_cod_data_X3D
from rotation_raw import RandomPolarRotation
from mm.functions.mm_filter import charpoly


def inverse_rotate_coords(x_prime, y_prime, angle_deg, center):
    theta = -angle_deg * math.pi / 180
    cx, cy = center

    x = cx + (x_prime - cx) * math.cos(theta) - (y_prime - cy) * math.sin(theta)
    y = cy + (x_prime - cx) * math.sin(theta) + (y_prime - cy) * math.cos(theta)
    return x, y


if __name__ == '__main__':

    base_dir = Path('/home/chris/Datasets/03_HORAO/CC_Rotation/')
    plot_opt = False
    save_opt = False
    skip_opt = False

    np.random.seed(3008)
    random.seed(3008)

    # load measured rotation angles and center points
    with open(base_dir / 'angles_and_center_points.txt', 'r') as f:
        lines = f.readlines()
    transforms = torch.tensor([[float(el) for el in line.strip('\n').split(' ')] for line in lines])

    # load calibration matrices
    calib_path = base_dir / '2021-10-21_C_2' / '550nm'
    A = read_cod_data_X3D(str(calib_path / '550_A.cod'))
    W = read_cod_data_X3D(str(calib_path / '550_W.cod'))
    pseudo_label = torch.ones([1, A.shape[0], A.shape[1]]) # create pseudo label to generate a mask (e.g., for bg removal)

    # instantiate models
    mm_model = MuellerMatrixSelector(mask_fun=None, norm_mueller=0, norm_opt=0, wnum=1)
    mueller_rotate = RandomPolarRotation(degrees=180, p=float('inf'))
    rotate = lambda x, angle, center: mueller_rotate.__call__(x, angle=angle, center=center, label=pseudo_label, transpose=True)
    if skip_opt: rotate = torchvision.transforms.functional.rotate

    # sort angle measurement folders
    from natsort import natsorted
    dir_list = natsorted([str(el) for el in base_dir.iterdir() if el.is_dir() and not str(el).__contains__('C_2')])

    # cyclic colormap for 180 degrees wrap-around
    cmap = plt.cm.twilight_shifted
    norm_uint8 = lambda x: ((x-x.min())/(x.max()-x.min()) * 255).astype(np.uint8) if (x.max()-x.min()) > 0 else (255*x).astype(np.uint8)

    # iterate through angle measurements
    ys, tps, fns, fps, tns, errs = [], [], [], [], [], []
    for i, dir in enumerate(dir_list):
        intensity = read_cod_data_X3D(Path(dir) / 'raw_data' / '550nm' / '550_Intensite.cod', raw_flag=True)
        bruit = read_cod_data_X3D(Path(dir) / 'raw_data' / '550nm' /'550_Bruit.cod', raw_flag=True)
        B = (intensity - bruit).moveaxis(-1, 0)
        F = torch.cat([B, A.moveaxis(-1, 0), W.moveaxis(-1, 0)], dim=0)

        random_idx = np.random.randint(0, i) if i > 0 else 0
        print(random_idx)
        
        t = sum([transforms[i-idx][0] for idx in range(random_idx)]) if random_idx > 0 else transforms[i][0]
        angle = float(t) if random_idx > 0 else 0
        center = transforms.mean(0)[1:].tolist() #t[1:].tolist()
        if angle != 0:
            print('s')
        result = rotate(F, angle=angle, center=center)
        f, m = result if not skip_opt else (result, rotate(pseudo_label, angle=angle, center=center))
        if skip_opt and True:
            # replace zeros with identity matrices
            f[16:32][:, f[16:32].sum(0) == 0] = torch.eye(4, dtype=f.dtype).flatten()[:, None, None].repeat(1, f.shape[1], f.shape[2])[:, f[16:32].sum(0) == 0]
            f[32:][:, f[32:].sum(0) == 0] = torch.eye(4, dtype=f.dtype).flatten()[:, None, None].repeat(1, f.shape[1], f.shape[2])[:, f[32:].sum(0) == 0]
        y = mm_model(f[None])

        # unrotated
        y_orig = mm_model(F[None])
        ys.append(y_orig)

        # coordinates transform
        h, w = y_orig.shape[-2:]
        center = (w / 2, h / 2)
        margin = 1

        for _ in range(100):
            xx_prime, yy_prime = random.randint(0+margin, w-1-margin), random.randint(0+margin, h-1-margin)  # random pixel coordinates
            xx, yy = inverse_rotate_coords(xx_prime, yy_prime, angle_deg=angle, center=center)
            #print(f"Output pixel ({xx_prime},{yy_prime}) maps to input pixel ({xx},{yy})")

            if 0+margin < xx < w-1-margin and 0+margin < yy < h-1-margin:
                # physical realizability assessment
                mm_y = y[..., int(round(yy)), int(round(xx))]
                mm_y_orig = y_orig[..., yy_prime, xx_prime]
                real_y = bool(charpoly(mm_y).squeeze().numpy())
                real_y_orig = bool(charpoly(mm_y_orig).squeeze().numpy())
                realizability_match = real_y == real_y_orig
                errs.append(realizability_match)
                if real_y is True and real_y_orig is True:
                    tps.append(True)
                elif real_y is False and real_y_orig is True:
                    fns.append(True)
                elif real_y is True and real_y_orig is False:
                    fps.append(True)
                elif real_y is False and real_y_orig is False:
                    tns.append(True)
            else:
                print('Mapped point coordinates are outside field of view')

        if i > 0 and random_idx > 0:
            y_diff = (y.sum(1)/y.sum(1).max()).squeeze()-(ys[-random_idx-1].sum(1)/ys[-random_idx-1].sum(1).max()).squeeze().numpy()
            y_img = (y.sum(1)/y.sum(1).max()).squeeze().numpy()
            if plot_opt:
                fig, axs = plt.subplots(1, 4, figsize=(15, 8))
                axs[0].imshow((y_orig.sum(1)/y_orig.sum(1).max()).squeeze(), vmin=0, vmax=1)
                axs[0].set_title('Input')
                axs[1].imshow(y_img, vmin=0, vmax=1)
                axs[1].set_title('Rotated')
                axs[2].imshow((ys[-random_idx-1].sum(1)/ys[-random_idx-1].sum(1).max()).squeeze(), vmin=0, vmax=1)
                axs[2].set_title('GT')
                axs[3].imshow(y_diff, vmin=0, vmax=1)
                axs[3].set_title('Difference')
                plt.tight_layout()
                plt.show()
            if save_opt:
                alpha = norm_uint8(m.squeeze().numpy())
                ext = ['_rect', '_wo'][skip_opt]
                imageio.imwrite('fig-energy-preserve-'+str(i).zfill(2)+ext+'.png', norm_uint8(np.stack((y_img, y_img, y_img, alpha), axis=-1)))
                imageio.imwrite('fig-energy-preserve-'+str(i).zfill(2)+'orig.png', norm_uint8(y_orig/y_orig.max()))
                imageio.imwrite('fig-energy-preserve-'+str(i).zfill(2)+'gt.png', norm_uint8(ys[-random_idx-1]/ys[-random_idx-1].max()))
                imageio.imwrite('fig-energy-preserve-'+str(i).zfill(2)+'diff.png', norm_uint8(np.stack((y_diff, y_diff, y_diff, alpha), axis=-1)))
        else:
            y_ref = y

        m_previous = m
        y_previous = y
    ratio = float(sum(errs)/len(errs)*100)
    print('Ratio for Mueller matrices remaining realizable: %s ' % str(round(ratio, 3)))
    print(sum(tps))
    print(sum(fns))
    print(sum(fps))
    print(sum(tns))
    print(sum(tps)/(sum(tps)+sum(fns))) if (sum(tps)+sum(fns)) > 0 else print('n.a.') # sensitivity
    print(sum(tns)/(sum(tns)+sum(fps))) if (sum(tns)+sum(fps)) > 0 else print('n.a.') # specificity
