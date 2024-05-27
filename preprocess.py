import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import SimpleITK as sitk
import torch
import warnings
import nibabel as nib
import numpy as np
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
from util import seed_everything

seed_everything(123456)
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')


def resample(resample_factor, ori_img, resample_method=sitk.sitkNearestNeighbor):
    ori_size = np.array(ori_img.GetSize())
    ori_spacing = np.array(ori_img.GetSpacing())
    tar_size = ori_size // resample_factor
    tar_spacing = ori_spacing * resample_factor
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)
    resampler.SetSize(tar_size.tolist())
    resampler.SetOutputSpacing(tar_spacing)
    if resample_method == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    resampler.SetOutputOrigin(ori_img.GetOrigin())
    resampler.SetOutputDirection(ori_img.GetDirection())
    resampled_img = resampler.Execute(ori_img)
    return resampled_img


def pad(img, padded_size):
    pad_val = np.min(img)
    if len(img.shape) == 3:
        N, H, W = img.shape
        N_tag = H_tag = W_tag = 0
        N_pad = padded_size - N
        if N_pad % 2 == 1:
            N_pad += 1
            N_tag = 1
        N_pad_size_left = N_pad_size_right = N_pad // 2
        H_pad = padded_size - H
        if H_pad % 2 == 1:
            H_pad += 1
            H_tag = 1
        H_pad_size_left = H_pad_size_right = H_pad // 2
        W_pad = padded_size - W
        if W_pad % 2 == 1:
            W_pad += 1
            W_tag = 1
        W_pad_size_left = W_pad_size_right = W_pad // 2
        img = np.pad(img, ((int(N_pad_size_left - N_tag), int(N_pad_size_right)),
                           (int(H_pad_size_left - H_tag), int(H_pad_size_right)),
                           (int(W_pad_size_left - W_tag), int(W_pad_size_right))), 'constant',
                     constant_values=((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val)))
    if len(img.shape) == 2:
        H, W = img.shape
        H_tag = W_tag = 0
        H_pad = padded_size - H
        if H_pad % 2 == 1:
            H_pad += 1
            H_tag = 1
        H_pad_size_left = H_pad_size_right = H_pad // 2
        W_pad = padded_size - W
        if W_pad % 2 == 1:
            W_pad += 1
            W_tag = 1
        W_pad_size_left = W_pad_size_right = W_pad // 2
        img = np.pad(img, ((int(H_pad_size_left - H_tag), int(H_pad_size_right)),
                           (int(W_pad_size_left - W_tag), int(W_pad_size_right))), 'constant',
                     constant_values=((pad_val, pad_val), (pad_val, pad_val)))
    return img


def cut(img_array, target_size):
    N, H, W = img_array.shape
    N_center = H_center = W_center = target_size // 2
    if N > target_size:
        if N % 2 == 1:
            N_center = (N + 1) // 2 - 1
        elif N % 2 == 0:
            N_center = N // 2
        N_left = N_center - target_size // 2
        N_right = N_center + target_size // 2
    else:
        N_left = 0
        N_right = N
    if H > target_size:
        if H % 2 == 1:
            H_center = (H + 1) // 2 - 1
        elif H % 2 == 0:
            H_center = H // 2
        H_left = H_center - target_size // 2
        H_right = H_center + target_size // 2
    else:
        H_left = 0
        H_right = H
    if W > target_size:
        if W % 2 == 1:
            W_center = (W + 1) // 2 - 1
        elif W % 2 == 0:
            W_center = W // 2
        W_left = W_center - target_size // 2
        W_right = W_center + target_size // 2
    else:
        W_left = 0
        W_right = W
    cut_img_array = img_array[N_left: N_right, H_left: H_right, W_left: W_right]
    return cut_img_array


def downsample_batch():
    resample_factor = 4
    dataset = 'train'
    img_type = 'raw'
    SRC_PATH = f'./data/dataset/VerSe/MASS/{dataset}/{img_type}'
    SAVE_PATH = f'./data/dataset/VerSe/MASS/{dataset}/{img_type}_128'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    img_files = os.listdir(SRC_PATH)
    for ct_item in tqdm(img_files):
        ct_path = SRC_PATH + '/' + ct_item
        resampled_img = sitk.ReadImage(ct_path)
        resampled_img_spacing = np.array(resampled_img.GetSpacing()) * resample_factor
        resampled_img_array = sitk.GetArrayFromImage(resampled_img)
        resampled_img_array = cut(resampled_img_array, target_size=512)
        resampled_img_array = pad(resampled_img_array, padded_size=512)
        resampled_img = sitk.GetImageFromArray(resampled_img_array)
        resampled_img = resample(resample_factor, resampled_img)
        resampled_img_array = sitk.GetArrayFromImage(resampled_img)
        resampled_img = sitk.GetImageFromArray(resampled_img_array)
        resampled_img.SetSpacing(resampled_img_spacing)
        sitk.WriteImage(resampled_img, SAVE_PATH + '/' + ct_item)


def read_nii(nii):
    return np.asanyarray(nib.load(nii).dataobj)


def save_slice(raw, mask, name, index, out_raw_dir, out_mask_dir):
    mask = mask > 0
    unique_values = np.unique(mask)
    if len(unique_values) > 1:
        out_raw_pt = f'{out_raw_dir}/{name}_{index}.png'
        im = Image.fromarray(raw)
        im = im.convert("L")
        im.save(out_raw_pt)

        out_mask_pt = f'{out_mask_dir}/{name}_{index}.png'
        im = Image.fromarray(mask)
        im = im.convert("L")
        im.save(out_mask_pt)


def is_diagonal(matrix):
    for i in range(0, 3):
        for j in range(0, 3):
            if ((i != j) and (matrix[i][j] != 0)):
                return False
    return True


def generate_data(raw_pt, mask_pt, name, out_raw_dir, out_mask_dir, skip_slice=3):
    continue_it = True
    raw = read_nii(raw_pt)
    mask = read_nii(mask_pt)

    if "split" in raw_pt:
        continue_it = False

    affine = nib.load(raw_pt).affine

    if is_diagonal(affine[:3, :3]):
        transposed_raw = np.transpose(raw, [2, 1, 0])
        transposed_raw = np.flip(transposed_raw)
        transposed_mask = np.transpose(mask, [2, 1, 0])
        transposed_mask = np.flip(transposed_mask)

    else:
        transposed_raw = np.rot90(raw)
        transposed_raw = np.flip(transposed_raw)

        transposed_mask = np.rot90(mask)
        transposed_mask = np.flip(transposed_mask)

    if continue_it:
        if transposed_raw.shape:
            slice_count = transposed_raw.shape[-1]
            print("File name: ", name, " - Slice count: ", slice_count)

            for each_slice in range(1, slice_count, skip_slice):
                save_slice(transposed_raw[:, :, each_slice],
                           transposed_mask[:, :, each_slice],
                           name,
                           each_slice,
                           out_raw_dir,
                           out_mask_dir)


def generate_slice():
    stage = 1
    if stage == 1:
        dir = './data/dataset/VerSe/MASS'
        train_raw_dir = './data/dataset/VerSe/MASS/train/raw'
        train_mask_dir = './data/dataset/VerSe/MASS/train/mask'
        val_raw_dir = './data/dataset/VerSe/MASS/val/raw'
        val_mask_dir = './data/dataset/VerSe/MASS/test/mask'
        test_raw_dir = './data/dataset/VerSe/MASS/test/raw'
        test_mask_dir = './data/dataset/VerSe/MASS/test/mask'

        train_raw_slice_dir = './data/dataset/VerSe/MASS/train/raw_slice'
        train_mask_slice_dir = './data/dataset/VerSe/MASS/train/mask_slice'
        val_raw_slice_dir = './data/dataset/VerSe/MASS/val/raw_slice'
        val_mask_slice_dir = './data/dataset/VerSe/MASS/val/mask_slice'
        test_raw_slice_dir = './data/dataset/VerSe/MASS/test/raw_slice'
        test_mask_slice_dir = './data/dataset/VerSe/MASS/test/mask_slice'

        if not os.path.exists(train_raw_slice_dir):
            os.makedirs(train_raw_slice_dir)
        if not os.path.exists(train_mask_slice_dir):
            os.makedirs(train_mask_slice_dir)
        if not os.path.exists(val_raw_slice_dir):
            os.makedirs(val_raw_slice_dir)
        if not os.path.exists(val_mask_slice_dir):
            os.makedirs(val_mask_slice_dir)
        if not os.path.exists(test_raw_slice_dir):
            os.makedirs(test_raw_slice_dir)
        if not os.path.exists(test_mask_slice_dir):
            os.makedirs(test_mask_slice_dir)

        train_set = os.listdir(train_raw_dir)
        val_set = os.listdir(val_raw_dir)
        test_set = os.listdir(test_raw_dir)
        # datasets = {'train': train_set, 'val': val_set, 'test': test_set}
        datasets = {'test': test_set}

        print(f'images count train: {len(train_set)}, validation: {len(val_set)}, test: {len(test_set)}')

        print("Processing started.")
        for dataset in datasets.keys():
            if dataset == 'train':
                skip_slice = 3
            else:
                skip_slice = 1
            for nii in tqdm(datasets[dataset]):
                name = nii.split('.nii.gz')[0]
                raw_pt = f'{dir}/{dataset}/raw/{nii}'
                mask_pt = f'{dir}/{dataset}/mask/{nii}'
                out_raw_dir = f'{dir}/{dataset}/raw_slice'
                out_mask_dir = f'{dir}/{dataset}/mask_slice'

                generate_data(raw_pt, mask_pt, name, out_raw_dir, out_mask_dir, skip_slice)
            print(f'Processing {dataset} data done.')
    elif stage == 2:
        dir = './data/dataset/MASS'
        all_raw_dir = './data/dataset/MASS/all/raw'
        all_mask_dir = './data/dataset/MASS/all/mask'

        all_raw_slice_dir = './data/dataset/MASS/all/raw_slice'
        all_mask_slice_dir = './data/dataset/MASS/all/mask_slice'
        train_raw_slice_dir = './data/dataset/MASS/train/raw_slice'
        train_mask_slice_dir = './data/dataset/MASS/train/mask_slice'
        val_raw_slice_dir = './data/dataset/MASS/val/raw_slice'
        val_mask_slice_dir = './data/dataset/MASS/val/mask_slice'
        test_raw_slice_dir = './data/dataset/MASS/test/raw_slice'
        test_mask_slice_dir = './data/dataset/MASS/test/mask_slice'

        if not os.path.exists(all_raw_slice_dir):
            os.makedirs(all_raw_slice_dir)
        if not os.path.exists(all_mask_slice_dir):
            os.makedirs(all_mask_slice_dir)
        if not os.path.exists(train_raw_slice_dir):
            os.makedirs(train_raw_slice_dir)
        if not os.path.exists(train_mask_slice_dir):
            os.makedirs(train_mask_slice_dir)
        if not os.path.exists(val_raw_slice_dir):
            os.makedirs(val_raw_slice_dir)
        if not os.path.exists(val_mask_slice_dir):
            os.makedirs(val_mask_slice_dir)
        if not os.path.exists(test_raw_slice_dir):
            os.makedirs(test_raw_slice_dir)
        if not os.path.exists(test_mask_slice_dir):
            os.makedirs(test_mask_slice_dir)

        all_set = os.listdir(all_raw_dir)

        print(f'images count all: {len(all_set)}')

        print("Processing started.")
        skip_slice = 1
        for nii in tqdm(all_set):
            if 'sub' in nii:
                name = nii.split('.nii.gz')[0]
                raw_pt = f'{dir}/all/raw/{nii}'
                mask_pt = f'{dir}/all/mask/{nii}'
                out_raw_dir = f'{dir}/all/raw_slice'
                out_mask_dir = f'{dir}/all/mask_slice'

                generate_data(raw_pt, mask_pt, name, out_raw_dir, out_mask_dir, skip_slice)
        print(f'Processing all data done.')

        all_raw_slices = os.listdir(all_raw_slice_dir)
        random.shuffle(all_raw_slices)

        train_set = all_raw_slices[:1063]
        val_set = all_raw_slices[1063:1196]
        test_set = all_raw_slices[1196:]

        for i in train_set:
            shutil.copyfile(f'{all_raw_slice_dir}/{i}', f'{train_raw_slice_dir}/{i}')
            shutil.copyfile(f'{all_mask_slice_dir}/{i}', f'{train_mask_slice_dir}/{i}')
        for i in val_set:
            shutil.copyfile(f'{all_raw_slice_dir}/{i}', f'{val_raw_slice_dir}/{i}')
            shutil.copyfile(f'{all_mask_slice_dir}/{i}', f'{val_mask_slice_dir}/{i}')
        for i in test_set:
            shutil.copyfile(f'{all_raw_slice_dir}/{i}', f'{test_raw_slice_dir}/{i}')
            shutil.copyfile(f'{all_mask_slice_dir}/{i}', f'{test_mask_slice_dir}/{i}')


if __name__ == "__main__":
    generate_slice()
