from pathlib import Path
import torch

from mm.models import MuellerMatrixModel
from mm.utils.cod import read_cod_data_X3D
from augmentations.rotation_raw import RandomPolarRotation

if __name__ == '__main__':

    base_dir = Path('/media/chris/EB62-383C/CC_Rotation/')
    plot_opt = False
    save_opt = True

    norm_uint8 = lambda x: ((x-x.min())/(x.max()-x.min()) * 255).astype(np.uint8)

    # load measured rotation angles and center points
    with open(base_dir / 'angles_and_center_points.txt', 'r') as f:
        lines = f.readlines()
    transforms = torch.tensor([[float(el) for el in line.strip('\n').split(' ')] for line in lines])

    # load calibration matrices
    calib_path = base_dir / '2021-10-21_C_2' / '550nm'
    A = read_cod_data_X3D(str(calib_path / '550_A.cod'))
    W = read_cod_data_X3D(str(calib_path / '550_W.cod'))

    # instantiate models
    feat_keys = ['azimuth']
    mm_model = MuellerMatrixModel(feature_keys=feat_keys, wnum=1)
    mueller_rotate = RandomPolarRotation(degrees=180, p=float('inf'))

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
        mueller_rotate.center = t[1:].tolist()
        pseudo_label = torch.ones_like(F[0][None]) # create pseudo label to generate a mask (e.g., for bg removal)
        F, m = mueller_rotate(F, label=pseudo_label, angle=angle, transpose=True)
        y = mm_model(F[None])

        if i > 0:
            if plot_opt:
                import matplotlib.pyplot as plt
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
                rgb = np.stack((y.squeeze().numpy(), y.squeeze().numpy(), y.squeeze().numpy()), axis=-1)
                rgb = norm_uint8(rgb)
                alpha = norm_uint8(m.squeeze().numpy())
                img = np.concatenate((rgb, alpha[..., None]), axis=-1)
                imageio.imwrite(str(i).zfill(2)+'.png', img)
        else:
            y_ref = y.clone()

        m_previous = m.clone()
        y_previous = y.clone()

    if save_opt:
        frames = []
        for fn in sorted(Path('.').glob('*.png')):
            print(fn.name)
            frames.append(imageio.v2.imread(fn))
        imageio.mimsave('./docs/animation_with_alpha.gif', frames, duration=len(frames)//2, loop=0, disposal=2)
        #imageio.mimsave('./docs/animation_with_alpha.webp', frames, format='WEBP', duration=len(frames)//2, quality=50, lossless=False)
