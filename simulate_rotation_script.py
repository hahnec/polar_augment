import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt

from mm.models import MuellerMatrixModel
from mm.utils.cod import read_cod_data_X3D
from augmentations.rotation_raw import RandomPolarRotation

if __name__ == '__main__':

    base_dir = Path('/media/chris/EB62-383C/CC_Rotation/')
    plot_opt = False
    save_opt = True
    gray_opt = False
    skip_opt = True

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
    mm_model = MuellerMatrixModel(feature_keys=['azimuth'], wnum=1)
    mueller_rotate = RandomPolarRotation(degrees=180, p=float('inf'))
    rotate = lambda x, angle, center: mueller_rotate.__call__(x, angle=angle, center=center, label=pseudo_label, transpose=True)
    if skip_opt: rotate = torchvision.transforms.functional.rotate

    # sort angle measurement folders
    from natsort import natsorted
    dir_list = natsorted([str(el) for el in base_dir.iterdir() if el.is_dir() and not str(el).__contains__('C_2')])

    # iterate through angle measurements
    for i, dir in enumerate(dir_list):
        intensity = read_cod_data_X3D(Path(dir) / 'raw_data' / '550nm' / '550_Intensite.cod', raw_flag=True)
        bruit = read_cod_data_X3D(Path(dir) / 'raw_data' / '550nm' /'550_Bruit.cod', raw_flag=True)
        B = (intensity - bruit).moveaxis(-1, 0)
        F = torch.cat([B, A.moveaxis(-1, 0), W.moveaxis(-1, 0)], dim=0)
        
        t = transforms[i]
        angle = angle + float(t[0]) if i > 0 else 0
        result = rotate(F, angle=angle, center=t[1:].tolist())
        F, m = result if not skip_opt else (result, rotate(pseudo_label, angle=angle, center=t[1:].tolist()))
        if skip_opt: 
            # replace zeros with identity matrices
            F[16:32][:, F[16:32].sum(0) == 0] = torch.eye(4, dtype=F.dtype).flatten()[:, None, None].repeat(1, F.shape[1], F.shape[2])[:, F[16:32].sum(0) == 0]
            F[32:][:, F[32:].sum(0) == 0] = torch.eye(4, dtype=F.dtype).flatten()[:, None, None].repeat(1, F.shape[1], F.shape[2])[:, F[32:].sum(0) == 0]
        y = mm_model(F[None])

        if i > 0:
            if plot_opt:
                fig, axs = plt.subplots(1, 3, figsize=(15, 8))
                axs[0].imshow(y_ref.squeeze().numpy())
                axs[0].set_title('Reference')
                axs[1].imshow(y_previous.squeeze().numpy(), alpha=m_previous.squeeze().numpy())
                axs[1].set_title('Previous')
                axs[2].imshow(y.squeeze().numpy(), alpha=m.squeeze().numpy())
                axs[2].set_title('Current')
                plt.show()
            if save_opt:
                import numpy as np
                import imageio
                if gray_opt:
                    rgb = np.stack((y.squeeze().numpy(), y.squeeze().numpy(), y.squeeze().numpy()), axis=-1)
                else:
                    # cyclic colormap for 180 degrees wrap-around
                    cmap = plt.cm.twilight_shifted
                    rgb = cmap((y/y.max()).squeeze().numpy())
                norm_uint8 = lambda x: ((x-x.min())/(x.max()-x.min()) * 255).astype(np.uint8) if (x.max()-x.min()) > 0 else (255*x).astype(np.uint8)
                rgb = norm_uint8(rgb)
                alpha = norm_uint8(m.squeeze().numpy())
                img = np.concatenate((rgb[..., :3], alpha[..., None]), axis=-1)
                ext = ['_rect', '_wo'][skip_opt]
                imageio.imwrite(str(i).zfill(2)+ext+'.png', img)
        else:
            y_ref = y.clone()

        m_previous = m.clone()
        y_previous = y.clone()

    if save_opt:
        frames = []
        for fn in sorted(Path('.').glob('*'+ext+'.png')):
            print(fn.name)
            frames.append(imageio.v2.imread(fn))
        imageio.mimsave('./docs/animation_with_alpha'+ext+'.gif', frames, duration=len(frames)//2, loop=0, disposal=2)
        #imageio.mimsave('./docs/animation_with_alpha.webp', frames, format='WEBP', duration=len(frames)//2, quality=50, lossless=False)
