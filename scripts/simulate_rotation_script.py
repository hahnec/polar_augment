import torch
import torchvision
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt

from mm.models import LuChipmanModel
from mm.utils.cod import read_cod_data_X3D
from rotation_raw import RandomPolarRotation
from padding import mirror_rotate

if __name__ == '__main__':

    base_dir = Path('/home/chris/Datasets/03_HORAO/CC_Rotation/')
    plot_opt = True
    save_opt = False
    skip_opt = False
    offset_method = False

    if offset_method: skip_opt = True

    np.random.seed(3008)

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
    mm_model = LuChipmanModel(feature_keys=['linr', 'azimuth', 'mask'], wnum=1, norm_mueller=1, norm_opt=0, filter_opt=False)
    mueller_rotate = RandomPolarRotation(degrees=180, p=float('inf'), pad_rotate=mirror_rotate)
    rotate = lambda x, angle, center: mueller_rotate.__call__(x, angle=angle, center=center, label=pseudo_label, transpose=True)
    if skip_opt: rotate = torchvision.transforms.functional.rotate

    # sort angle measurement folders
    from natsort import natsorted
    dir_list = natsorted([str(el) for el in base_dir.iterdir() if el.is_dir() and not str(el).__contains__('C_2')])

    # cyclic colormap for 180 degrees wrap-around
    cmap = plt.cm.twilight_shifted
    norm_uint8 = lambda x: ((x-x.min())/(x.max()-x.min()) * 255).astype(np.uint8) if (x.max()-x.min()) > 0 else (255*x).astype(np.uint8)

    # iterate through angle measurements
    ys, vs, ls, errs = [], [], [], []
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
        result = rotate(F, angle=angle, center=center)
        f, m = result if not skip_opt else (result, rotate(pseudo_label, angle=angle, center=center))
        if skip_opt: 
            # replace zeros with identity matrices
            f[16:32][:, f[16:32].sum(0) == 0] = torch.eye(4, dtype=f.dtype).flatten()[:, None, None].repeat(1, f.shape[1], f.shape[2])[:, f[16:32].sum(0) == 0]
            f[32:][:, f[32:].sum(0) == 0] = torch.eye(4, dtype=f.dtype).flatten()[:, None, None].repeat(1, f.shape[1], f.shape[2])[:, f[32:].sum(0) == 0]
        y = mm_model(f[None])
        l, y, v = y[:, 0][:, None], y[:, 1][:, None], y[:, 2][:, None]

        if offset_method and angle != 0: 
            y = (y + angle) % 180
            y[y<0] = 180 + y[y<0]

        rgb = cmap((y/y.max()).squeeze().numpy())

        # unrotated
        y_orig = mm_model(F[None]).squeeze().numpy()
        l_orig, y_orig, v_orig = y_orig[0], y_orig[1], y_orig[2]
        #y_orig = cmap((y_orig/y_orig.max()))
        ys.append(y_orig)
        vs.append(v_orig)
        ls.append(l_orig)

        if i > 0 and random_idx > 0:
            mask = (m*v*vs[-random_idx-1]).squeeze().numpy()
            if 'linr' in mm_model.feature_keys: mask = mask * (ls[-random_idx-1] > np.percentile(ls[-random_idx-1], 75))
            yd1 = (180-y.squeeze().numpy()) - ys[-random_idx-1]
            yd2 = y.squeeze().numpy() - ys[-random_idx-1]
            diff = np.minimum(abs(yd1), abs(yd2))[mask.astype(bool)]
            errs.extend(diff)
            if plot_opt:
                fig, axs = plt.subplots(1, 4, figsize=(15, 8))
                axs[0].imshow(cmap(y_orig/y_orig.max()).squeeze())
                axs[0].set_title('Input')
                axs[1].imshow(rgb[..., :3], alpha=m.squeeze().numpy())
                axs[1].set_title('Rotated')
                axs[2].imshow(cmap(ys[-random_idx-1]/ys[-random_idx-1].max()))
                axs[2].set_title('GT')
                axs[3].imshow(mask)
                axs[3].set_title('Mask')
                plt.show()
            if save_opt:
                rgb = norm_uint8(rgb)
                alpha = norm_uint8(m.squeeze().numpy())
                img = np.concatenate((rgb[..., :3], alpha[..., None]), axis=-1)
                ext = ['_rect', '_wo'][skip_opt]
                imageio.imwrite('fig-'+str(i).zfill(2)+ext+'.png', img)
                imageio.imwrite('fig-'+str(i).zfill(2)+'_orig.png', norm_uint8(cmap(y_orig/y_orig.max())))
                imageio.imwrite('fig-'+str(i).zfill(2)+'_gt.png', norm_uint8(cmap(ys[-random_idx-1]/ys[-random_idx-1].max())))
                imageio.imwrite('fig-'+str(i).zfill(2)+'_diff.png', norm_uint8(np.stack((diff, diff, diff, alpha), axis=-1)))
        else:
            y_ref = y

        m_previous = m
        y_previous = y

    print(np.round(np.mean(np.array(errs)**2)**.5, 4))
    print(np.round(np.mean(abs(np.array(errs))), 4))