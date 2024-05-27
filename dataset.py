import os
import random
import shutil

import SimpleITK as sitk
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from util import seed_everything

seed_everything(123456)


def split_dataset():
    raw_dir = './data/dataset/VerSe/MASS/all/raw'
    mask_dir = './data/dataset/VerSe/MASS/all/mask'
    masks = os.listdir(mask_dir)
    random.shuffle(masks)
    train_set = masks[:285]
    val_set = masks[285:320]
    test_set = masks[320:]
    if not os.path.exists('./data/dataset/VerSe/MASS/train/raw'):
        os.makedirs('./data/dataset/VerSe/MASS/train/raw')
    if not os.path.exists('./data/dataset/VerSe/MASS/train/mask'):
        os.makedirs('./data/dataset/VerSe/MASS/train/mask')
    if not os.path.exists('./data/dataset/VerSe/MASS/val/raw'):
        os.makedirs('./data/dataset/VerSe/MASS/val/raw')
    if not os.path.exists('./data/dataset/VerSe/MASS/val/mask'):
        os.makedirs('./data/dataset/VerSe/MASS/val/mask')
    if not os.path.exists('./data/dataset/VerSe/MASS/test/raw'):
        os.makedirs('./data/dataset/VerSe/MASS/test/raw')
    if not os.path.exists('./data/dataset/VerSe/MASS/test/mask'):
        os.makedirs('./data/dataset/VerSe/MASS/test/mask')
    for i in train_set:
        shutil.copyfile(f'{raw_dir}/{i}', f'./data/dataset/VerSe/MASS/train/raw/{i}')
        shutil.copyfile(f'{mask_dir}/{i}', f'./data/dataset/VerSe/MASS/train/mask/{i}')
    for i in val_set:
        shutil.copyfile(f'{raw_dir}/{i}', f'./data/dataset/VerSe/MASS/val/raw/{i}')
        shutil.copyfile(f'{mask_dir}/{i}', f'./data/dataset/VerSe/MASS/val/mask/{i}')
    for i in test_set:
        shutil.copyfile(f'{raw_dir}/{i}', f'./data/dataset/VerSe/MASS/test/raw/{i}')
        shutil.copyfile(f'{mask_dir}/{i}', f'./data/dataset/VerSe/MASS/test/mask/{i}')


def generate_verse():
    dir = './data/dataset/VerSe'
    save_dir = './data/dataset/VerSe/MASS'
    if not os.path.exists(f'{save_dir}/raw'):
        os.makedirs(f'{save_dir}/raw')
    if not os.path.exists(f'{save_dir}/mask'):
        os.makedirs(f'{save_dir}/mask')
    verses = os.listdir(dir)
    verses.remove('MASS')
    for i in verses:
        if os.path.isdir(f'{dir}/{i}'):
            raws = os.listdir(f'{dir}/{i}/{i}/rawdata')
            for j in raws:
                if os.path.isdir(f'{dir}/{i}/{i}/rawdata/{j}'):
                    raw_dir = f'{dir}/{i}/{i}/rawdata/{j}'
                    mask_dir = f'{dir}/{i}/{i}/derivatives/{j}'
                    raw_files = os.listdir(raw_dir)
                    mask_files = os.listdir(mask_dir)
                    for k in raw_files:
                        if k.endswith('.nii.gz'):
                            shutil.copyfile(f'{raw_dir}/{k}', f'{save_dir}/raw/{k}')
                    for k in mask_files:
                        if k.endswith('.nii.gz'):
                            shutil.copyfile(f'{mask_dir}/{k}', f'{save_dir}/mask/{k}')


def change_verse_name():
    raw_dir = './data/dataset/VerSe/MASS/raw'
    mask_dir = './data/dataset/VerSe/MASS/mask'
    raws = os.listdir(raw_dir)
    masks = os.listdir(mask_dir)
    for i in raws:
        shutil.move(f'{raw_dir}/{i}', f'{raw_dir}/{i.split(".nii.gz")[0]}.nii.gz')
    for i in masks:
        shutil.move(f'{mask_dir}/{i}', f'{mask_dir}/{i.split(".nii.gz")[0]}.nii.gz')


class Seg3DDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, images):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.raw_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        image = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(mask)

        c, w, h = image.shape
        image = image.reshape((1, c, w, h))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        mask = mask.reshape((1, c, w, h))
        return {
            'name': f'{self.images[index].split(".nii.gz")[0]}',
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }


class Seg2DDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, images, w=256, h=256):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.images = images
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.raw_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        w, h = image.size
        image = image.resize((w, h), resample=Image.BICUBIC)
        image = T.Resize(size=(self.w, self.h))(image)
        image_ndarray = np.asarray(image)
        image_ndarray = image_ndarray.reshape(1, image_ndarray.shape[0], image_ndarray.shape[1])

        mask = mask.resize((w, h), resample=Image.NEAREST)
        mask = T.Resize(size=(self.w, self.h))(mask)
        mask_ndarray = np.asarray(mask)

        return {
            'name': f'{self.images[index].split(".png")[0]}',
            'image': torch.as_tensor(image_ndarray.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_ndarray.copy()).float().contiguous()
        }


# if __name__ == '__main__':
#     split_dataset()
