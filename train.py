import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import warnings
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Seg2DDataset
from module import UNet, AttUNet, ResUnet, MASSNet, UnetPlusPlus, U2NET, U2NETP
from util import DSCLoss, DSC, FocalLoss, IoULoss, IoU, Precision, Recall, Accuracy, BCELoss, FocalTverskyLoss
from util import seed_everything

warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': '{:.4f}'.format})

seed = 123456
seed_everything(seed)

lr = 1e-4
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'massnet'
save_name = 'massnet_1_1_1_1_7_3'
save_dir = f'./data/model/{save_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

stage = 2
resume = True
resume_epoch = -1
if resume:
    models = os.listdir(save_dir)
    resume_epoch = len(models) - 1
    # resume_epoch = 13

if stage == 1:
    epochs = 15 - resume_epoch - 1
elif stage == 2:
    epochs = 20 - resume_epoch - 1


def train():
    if stage == 1:
        train_raw_dir = f'./data/dataset/VerSe/MASS/train/raw_slice'
        train_mask_dir = f'./data/dataset/VerSe/MASS/train/mask_slice'
        val_raw_dir = f'./data/dataset/VerSe/MASS/val/raw_slice'
        val_mask_dir = f'./data/dataset/VerSe/MASS/val/mask_slice'
    elif stage == 2:
        train_raw_dir = f'./data/dataset/MASS/train/raw_slice'
        train_mask_dir = f'./data/dataset/MASS/train/mask_slice'
        val_raw_dir = f'./data/dataset/MASS/val/raw_slice'
        val_mask_dir = f'./data/dataset/MASS/val/mask_slice'


    train_images = os.listdir(train_raw_dir)
    val_images = os.listdir(val_raw_dir)

    train_dataset = Seg2DDataset(train_raw_dir, train_mask_dir, train_images)
    valid_dataset = Seg2DDataset(val_raw_dir, val_mask_dir, val_images)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # model = MASSNet(use_mar=(stage==2))
    model = MASSNet(use_mar=True)
    # model = UnetPlusPlus()
    model.to(device=device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr * 1e-2, max_lr=lr, step_size_up=100, cycle_momentum=False)
    if resume:
        ckpt = torch.load(f'{save_dir}/{model_name}{resume_epoch}.pth')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f'resume_model: {save_dir}/{model_name}{resume_epoch}')

    train_loss = []
    val_loss = []
    for epoch in range(resume_epoch + 1, resume_epoch + 1 + epochs):

        model.seg_net.deep_supervision = True
        model.train()
        train_running_loss = 0.0
        counter = 0

        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{resume_epoch + 1 + epochs}', unit='img') as pbar:
            for batch in train_dataloader:
                counter += 1
                # idx = random.randint(0, 192)
                # image = batch['image'].to(device)[:, :, idx:idx + 64, :, :]
                # mask = batch['mask'].to(device)[:, :, idx:idx + 64, :, :]
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)

                optimizer.zero_grad()
                outputs = model(image)

                # loss = 0.1 * (0.7 * FocalLoss(outputs[0].squeeze(), mask) + 0.3 * FocalTverskyLoss(outputs[0].squeeze(), mask)) \
                #     + 0.2 * (0.7 * FocalLoss(outputs[1].squeeze(), mask) + 0.3 * FocalTverskyLoss(outputs[1].squeeze(), mask)) \
                #     + 0.3 * (0.7 * FocalLoss(outputs[2].squeeze(), mask) + 0.3 * FocalTverskyLoss(outputs[2].squeeze(), mask)) \
                #     + 0.4 * (0.7 * FocalLoss(outputs[3].squeeze(), mask) + 0.3 * FocalTverskyLoss(outputs[3].squeeze(), mask)) \

                loss = 0.25 * (0.7 * BCELoss(outputs[0].squeeze(), mask) + 0.3 * DSCLoss(outputs[0].squeeze(), mask)) \
                    + 0.25 * (0.7 * BCELoss(outputs[1].squeeze(), mask) + 0.3 * DSCLoss(outputs[1].squeeze(), mask)) \
                    + 0.25 * (0.7 * BCELoss(outputs[2].squeeze(), mask) + 0.3 * DSCLoss(outputs[2].squeeze(), mask)) \
                    + 0.25 * (0.7 * BCELoss(outputs[3].squeeze(), mask) + 0.3 * DSCLoss(outputs[3].squeeze(), mask)) \

                # loss = 0.5 * BCELoss(outputs.squeeze(), mask) + 0.5 * DSCLoss(outputs.squeeze(), mask)

                train_running_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            train_loss.append(float(f'{train_running_loss / counter:.3f}'))
            print(train_loss)

        model.seg_net.deep_supervision = False
        model.eval()
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                counter += 1

                image = data['image'].to(device)
                mask = data['mask'].to(device)

                outputs = model(image)

                loss = DSC(outputs.squeeze(), mask)
                valid_running_loss += loss.item()

            val_loss.append(float(f'{valid_running_loss / counter:.3f}'))
            print(val_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{save_dir}/{model_name}{epoch}.pth')


if __name__ == '__main__':
    train()
