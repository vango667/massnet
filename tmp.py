import os

import SimpleITK as sitk
from tqdm import tqdm


def test():
    dir = './data/dataset/MASS/all/raw'
    cts = os.listdir(dir)
    lens = []
    for i in tqdm(cts):
        ct = sitk.ReadImage(f'{dir}/{i}')
        ct = sitk.GetArrayFromImage(ct)
        lens.append(ct.shape[0])
    print(max(lens))


if __name__ == '__main__':
    test()
