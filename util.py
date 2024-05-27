import os
import random

import numpy as np

import torch
import torch.nn as nn
from scipy import ndimage
import GeodisTK
from scipy.spatial.distance import directed_hausdorff

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Accuracy(inputs, targets, smooth=1e-5):
    inputs = inputs > 0.5
    targets = targets >= 1
    tp = (inputs & targets).sum()
    tn = (~inputs & ~targets).sum()
    fp = (inputs & ~targets).sum()
    fn = (~inputs & targets).sum()
    precision = (tp + tn + smooth) / (tp + fp + tn + fn + smooth)
    return precision


def Precision(inputs, targets, smooth=1e-5):
    inputs = inputs > 0.5
    targets = targets >= 1
    tp = (inputs & targets).sum()
    fp = (inputs & ~targets).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision


def Recall(inputs, targets, smooth=1e-5):
    inputs = inputs > 0.5
    targets = targets >= 1
    tp = (inputs & targets).sum()
    fn = (~inputs & targets).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def DSC(inputs, targets, smooth=1e-5):
    targets = targets >= 1
    targets = targets.float()
    intersection = (inputs * targets).sum()
    dsc = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dsc


def DSCLoss(inputs, targets):
    dsc_loss = 1 - DSC(inputs, targets)
    return dsc_loss


def IoU(inputs, targets, smooth=1e-5):
    targets = targets >= 1
    targets = targets.float()
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum() - intersection.sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou


def IoULoss(inputs, targets):
    iou_loss = 1 - IoU(inputs, targets)
    return iou_loss


def BCELoss(inputs, targets):
    targets = targets >= 1
    targets = targets.float()
    bce_loss = nn.BCELoss()(inputs, targets)
    return bce_loss


class FocalLossClass(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLossClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        bce_loss = BCELoss(inputs, targets)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def FocalLoss(inputs, targets):
    targets = targets >= 1
    targets = targets.float()
    focal_loss = FocalLossClass()(inputs, targets)
    return focal_loss


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class TverskyLossClass(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLossClass, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky


def TverskyLoss(inputs, targets):
    targets = targets >= 1
    targets = targets.float()
    focal_tversky_loss = TverskyLossClass()(inputs, targets)
    return focal_tversky_loss


class FocalTverskyLossClass(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """

    def __init__(self, gamma=0.75):
        super(FocalTverskyLossClass, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLossClass()

    def forward(self, net_output, target):
        tversky_loss = 1 + self.tversky(net_output, target)  # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


def FocalTverskyLoss(inputs, targets):
    targets = targets >= 1
    targets = targets.float()
    focal_tversky_loss = FocalTverskyLossClass()(inputs, targets)
    return focal_tversky_loss


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if dim == 2:
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def HD95(inputs, targets, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        inputs: a 3D or 2D binary image for segmentation
        targets: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    targets = targets >= 1
    targets = targets.float()

    s_edge = get_edge_points(inputs)
    g_edge = get_edge_points(targets)
    image_dim = len(inputs.shape)
    assert image_dim == len(targets.shape)
    if spacing == None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim == len(spacing)
    img = np.zeros_like(inputs)
    if image_dim == 2:
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def ASSD(inputs, targets, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        inputs: a 3D or 2D binary image for segmentation
        targets: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    targets = targets >= 1
    targets = targets.float()

    s_edge = get_edge_points(inputs)
    g_edge = get_edge_points(targets)
    image_dim = len(inputs.shape)
    assert image_dim == len(targets.shape)
    if spacing == None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim == len(spacing)
    img = np.zeros_like(inputs)
    if image_dim == 2:
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


def HD(inputs, targets, spacing=None):
    hd = directed_hausdorff(inputs, targets)[0]
    return hd


if __name__ == '__main__':
    x = torch.tensor([[[[0.96, 0.96, 0.96], [0.96, 0.8, 0.96], [0.96, 0.96, 0.7]]]], dtype=torch.float)
    y = torch.tensor([[[[2, 1, 1], [1, 4, 1], [1, 1, 6]]]], dtype=torch.float)
    print(x.shape, y.shape)
    print(HD(x.squeeze(), y.squeeze()))
