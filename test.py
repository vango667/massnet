import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import warnings
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Seg2DDataset
from module import UNet, AttUNet, ResUnet, UnetPlusPlus, MASSNet
from util import DSCLoss, DSC, FocalLoss, IoULoss, IoU, Precision, Recall, Accuracy, BCELoss, FocalTverskyLoss, HD95, ASSD, HD
from util import seed_everything
import matplotlib.pyplot as plt
import cv2


warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': '{:.4f}'.format})

seed = 123456
seed_everything(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

# model_name = 'massnet'
# save_name = 'massnet_stage1'
model_name = 'resunet'
save_name = 'resunet'
save_dir = f'./data/model/{save_name}'
models = os.listdir(save_dir)
resume_epoch = len(models) - 1
# resume_epoch = 17
print(f'resume_model: {save_dir}/{model_name}{resume_epoch}')

stage = 2


def test(resume_epoch):
    torch.cuda.empty_cache()

    if stage == 1:
        test_raw_dir = './data/dataset/VerSe/MASS/test/raw_slice'
        test_mask_dir = './data/dataset/VerSe/MASS/test/mask_slice'
    elif stage == 2:
        test_raw_dir = './data/dataset/MASS/test/raw_slice'
        test_mask_dir = './data/dataset/MASS/test/mask_slice'

    test_images = os.listdir(test_raw_dir)

    test_dataset = Seg2DDataset(test_raw_dir, test_mask_dir, test_images)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_name == 'unet':
        model = UNet()
    elif model_name == 'attunet':
        model = AttUNet()
    elif model_name == 'resunet':
        model = ResUnet()
    elif model_name == 'massnet':
        model = MASSNet(deep_supervision=False, use_mar=False)
    else:
        model = MASSNet(deep_supervision=True)
    model.to(device=device)
    ckpt = torch.load(f'{save_dir}/{model_name}{resume_epoch}.pth')
    model.load_state_dict(ckpt['model_state_dict'])

    test_acc = []
    test_p = []
    test_r = []
    test_dsc = []
    test_iou = []
    test_hd = []
    # test_hd95 = []
    # test_assd = []

    model.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            # print(data['name'])
            image = data['image'].to(device)
            mask = data['mask'].to(device)

            # x, delta_x, outputs = model(image)
            outputs = model(image)
            # outputs = 0.1 * outputs[0] + 0.2 * outputs[1] + 0.3 * outputs[2] + 0.4 * outputs[3]
            outputs = outputs.squeeze().cpu()
            mask = mask.squeeze().cpu()

            # x_1 = image.cpu().detach().squeeze().numpy()
            # plt.imshow(x_1, cmap='gray')
            # plt.show()
            # delta_x = delta_x.cpu().detach().squeeze().numpy()
            # plt.imshow(delta_x, cmap='gray')
            # plt.show()
            # x_2 = x.cpu().detach().squeeze().numpy()
            # plt.imshow(x_2, cmap='gray')
            # plt.show()
            z = outputs.cpu().detach().squeeze().numpy()
            # plt.imshow(z, cmap='gray')
            # plt.show()
            # y = mask.cpu().detach().squeeze().numpy()
            # plt.imshow(z, cmap='gray')
            # plt.show()
            # z_nomar = outputs.cpu().detach().squeeze().numpy()
            # plt.imshow(outputs, cmap='gray')
            # plt.show()
            # cv2.imwrite(f'./data/result/{data["name"][0]}_input.png', x_1)
            # cv2.imwrite(f'./data/result/{data["name"][0]}_ma.png', delta_x)
            # cv2.imwrite(f'./data/result/{data["name"][0]}_mar.png', x_2 * 255)
            # cv2.imwrite(f'./data/result/{data["name"][0]}_output.png', z * 255)
            cv2.imwrite(f'./data/result/{data["name"][0]}_resunet.png', z * 255)
            # cv2.imwrite(f'./data/result/{data["name"][0]}_y.png', y)
            # cv2.imwrite(f'./data/result/{data["name"][0]}_output_nomar.png', z_nomar * 255)

            acc = Accuracy(outputs, mask)
            p = Precision(outputs, mask)
            r = Recall(outputs, mask)
            dsc = DSC(outputs, mask)
            iou = IoU(outputs, mask)
            hd = HD(outputs, mask)
            # hd95 = HD95(outputs, mask)
            # assd = ASSD(outputs, mask)

            test_acc.append(acc.item())
            test_p.append(p.item())
            test_r.append(r.item())
            test_dsc.append(dsc.item())
            test_iou.append(iou.item())
            test_hd.append(hd)
            # test_hd95.append(hd95.item())
            # test_assd.append(assd.item())

            # outputs[outputs<=0.5] = 0
            # outputs[outputs>0.5] = 1.0
            # res = outputs.squeeze().cpu().numpy()
            # res = sitk.GetImageFromArray(res)
            # sitk.WriteImage(res, f'{data["name"][0]}_res.nii.gz')

            # tar = mask.squeeze().cpu().numpy()
            # tar = sitk.GetImageFromArray(tar)
            # sitk.WriteImage(tar, f'{data["name"][0]}_tar.nii.gz')

            # break
        test_acc = np.array(test_acc)
        test_p = np.array(test_p)
        test_r = np.array(test_r)
        test_dsc = np.array(test_dsc)
        test_iou = np.array(test_iou)
        test_hd = np.array(test_hd)
        # test_hd95 = np.array(test_hd95)
        # test_assd = np.array(test_assd)


        print(f'Accuracy: {np.mean(test_acc):.3f}±{np.std(test_acc):.3f}')
        print(f'Precision: {np.mean(test_p):.3f}±{np.std(test_p):.3f}')
        print(f'Recall: {np.mean(test_r):.3f}±{np.std(test_r):.3f}')
        print(f'DSC: {np.mean(test_dsc):.3f}±{np.std(test_dsc):.3f}')
        print(f'IoU: {np.mean(test_iou):.3f}±{np.std(test_iou):.3f}')
        print(f'HD: {np.mean(test_hd):.3f}±{np.std(test_hd):.3f}')
        # print(f'HD-95: {np.mean(test_hd95)}±{np.std(test_hd95)}')
        # print(f'ASSD: {np.mean(test_assd)}±{np.std(test_assd)}')


if __name__ == '__main__':
    for i in range(19, 20):
        test(i)
