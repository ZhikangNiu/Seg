# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 19:50
# @Author  : Zhikang Niu
# @FileName: train.py
# @Software: PyCharm
import logging
import random
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import dataloader
import torch.distributed as dist
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import utils
import config
from torch.cuda.amp import autocast
from models import PVTv2_Lawin
from losses import compute_loss, DiceLoss, SegmentationLoss
from datasets import SegData
from metrics import Metrics
from torchvision.transforms import transforms
from augmentations import Mixup_transmix

from sklearn.model_selection import KFold


val_best_fwiou = 0
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

opt = config.get_options()

# deveice init
torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(backend='nccl')

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# transform = transforms.Compose([
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomHorizontalFlip(p=0.5),
#     # transforms.RandomRotation(90),
# ])

train_dataset = SegData(opt.data_root, split='train', transform=None)
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# # do not shuffle
# train_dataloader = dataloader.DataLoader(
#     dataset=train_dataset,
#     batch_size=opt.batch_size//4,
#     num_workers=opt.workers,
#     pin_memory=True,
#     shuffle=False,
#     sampler=train_sampler,
#     drop_last=True,
# )

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(f"train_PVTv2_Lawin_B3_bs{opt.batch_size}_lr{opt.lr}.log")
formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
file_handler.setFormatter(formatter)

# print to screen
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# models init
model = PVTv2_Lawin('B3', 9, pretrained=opt.pretrained)

if opt.resume:
    state_dict = torch.load("./Best_PVTv2_Lawin_bs8_8_11.pth", map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("---------- load best pretrained model ----------")

model = model.cuda()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=False)

# criterion init
blob_loss_dict = {
    "main_weight": 2,
    "blob_weight": 1,
}
criterion_dict = {
    "ce": {
        "name": "ce",
        "loss": nn.CrossEntropyLoss(reduction="mean"),
        "weight": 1.0,
        "sigmoid": True,
    },
    "dice": {
        "name": "dice",
        "loss": DiceLoss(),
        "weight": 1.0,
        "sigmoid": True,
    },
}
blob_criterion_dict = {
    "ce": {
        "name": "ce",
        "loss": nn.CrossEntropyLoss(reduction="mean"),
        "weight": 1.0,
        "sigmoid": True,
    },
    "dice": {
        "name": "dice",
        "loss": DiceLoss(),
        "weight": 1.0,
        "sigmoid": True,
    },
}

metric = Metrics(num_classes=9, ignore_label=255, device="cuda")

# mixup_args = dict(
#     mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
#     prob=1.0, switch_prob=0.8, mode='batch',
#     label_smoothing=0.1, num_classes=9
# )
# mixup_fn = Mixup_transmix(**mixup_args)
# mixup_fn.mixup_enabled = True

# optim and scheduler init
params = [p for p in model.parameters() if p.requires_grad]
model_optimizer = optim.AdamW(params, lr=opt.lr, eps=1e-6, weight_decay=1e-5)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

criterion = SegmentationLoss(cuda=opt.cuda).build_loss(mode='ce')

if dist.get_rank() == 0:
    logger.info("-----------------train-----------------")

kfold = KFold(n_splits=10,shuffle=True,random_state=42)
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    if dist.get_rank() == 0:
        logger.info(f'---------------Fold: {fold}-----------------')
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size//4,
        sampler=train_subsampler
    )
    val_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=test_subsampler
    )

    train_length = len(train_dataset)

    for epoch in range(opt.niter):
        model.train()
        epoch_losses = utils.AverageMeter()
        epoch_iou = utils.AverageMeter()
        epoch_f1 = utils.AverageMeter()
        epoch_acc = utils.AverageMeter()
        epoch_fwiou = utils.AverageMeter()

        with tqdm(total=(train_length - train_length % opt.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

            for inputs, labels in train_dataloader:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # inputs, labels_mix = mixup_fn(inputs, labels)

                model_optimizer.zero_grad()

                out = model(inputs)

                # for attn in attns:
                #     attn = torch.mean(attn[:, :, 0, :], dim=1)# attn from cls_token to images
                #     print(attn.shape)
                #     labels_mix_calc_loss = mixup_fn.transmix_label(labels_mix, attn, inputs.shape)

                # loss, _, _ = compute_loss(blob_loss_dict, criterion_dict, blob_criterion_dict, out, labels, labels)
                loss = criterion(out, labels)
                loss.backward()
                model_optimizer.step()

                metric.update(out, labels)
                _, iou = metric.compute_iou()
                _, f1 = metric.compute_f1()
                _, acc = metric.compute_pixel_acc()
                fwiou = metric.compute_Frequency_Weighted_Intersection_over_Union()

                epoch_losses.update(loss.item(), opt.batch_size)
                epoch_iou.update(iou, opt.batch_size)
                epoch_f1.update(f1, opt.batch_size)
                epoch_acc.update(acc, opt.batch_size)
                epoch_fwiou.update(fwiou * 100, opt.batch_size)

                t.set_postfix(
                    loss='{:.6f}'.format(epoch_losses.avg),
                    iou='{:.2f}'.format(epoch_iou.avg),
                    f1='{:.2f}'.format(epoch_f1.avg),
                    acc='{:.2f}'.format(epoch_acc.avg),
                    fwiou='{:.2f}'.format(epoch_fwiou.avg),
                )
                t.update(opt.batch_size)
            if dist.get_rank() == 0:
                logger.info(f"| Fold:{fold} | epoch:{epoch} | loss:{epoch_losses.avg} | iou:{epoch_iou.avg} | f1:{epoch_f1.avg} | acc:{epoch_acc.avg} |")

        model_scheduler.step()

        model.eval()
        with torch.no_grad():
            val_iou_epoch = utils.AverageMeter()
            val_f1_epoch = utils.AverageMeter()
            val_acc_epoch = utils.AverageMeter()
            val_fwiou_epoch = utils.AverageMeter()
            for images, labels in tqdm(val_dataloader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)
                metric.update(output, labels)

                _, val_iou = metric.compute_iou()
                _, val_f1 = metric.compute_f1()
                _, val_acc = metric.compute_pixel_acc()
                val_fwiou = metric.compute_Frequency_Weighted_Intersection_over_Union()

                val_iou_epoch.update(iou)
                val_f1_epoch.update(f1)
                val_acc_epoch.update(acc)
                val_fwiou_epoch.update(fwiou * 100)

        if dist.get_rank() == 0:
            logger.info('val_iou: {:.2f}'.format(val_iou_epoch.avg))
            logger.info('val_f1: {:.2f}'.format(val_f1_epoch.avg))
            logger.info('val_acc: {:.2f}'.format(val_acc_epoch.avg))
            logger.info('val_fwiou: {:.2f}'.format(val_fwiou_epoch.avg))

        if val_fwiou_epoch.avg > val_best_fwiou:
            val_best_fwiou = val_fwiou_epoch.avg
            torch.save(model.state_dict(), f"./Best_PVTv2_b3_Lawin_bs{opt.batch_size}_{fold}_{epoch}.pth")

if dist.get_rank() == 0:
    torch.save(model.state_dict(), "./Last_PVTv2_Lawin.pth")
