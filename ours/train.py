# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 19:50
# @Author  : Zhikang Niu
# @FileName: train.py
# @Software: PyCharm
import random
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
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
from models import PVTv2_SegFormer
from losses import Dice,SegmentationLoss
from datasets import SegData
from metrics import Metrics

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


# transform = {
#     'image': A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.Normalize([0.24138375, 0.25477552, 0.29299292],
#                     [0.09506353, 0.09248942, 0.09274331]),
#         ToTensorV2(),
#         ]),
#     'label': A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         ToTensorV2()
#         ]),
# }

train_dataset = SegData(opt.data_root, split='train', transform=None)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# do not shuffle
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size//4,
    num_workers=opt.workers,
    pin_memory=True,
    shuffle=False,
    sampler=train_sampler,
    drop_last=True,
)


length = len(train_dataset)

# models init
model = PVTv2_SegFormer('B1', 9,pretrained=opt.pretrained).cuda()

if opt.resume:
    state_dict = torch.load("./Best_PVTv2_SegFormer.pth")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("---------- load best pretrained model ----------")

model = DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=False)


# criterion init
#criterion = Dice()
criterion = SegmentationLoss(cuda=opt.cuda).build_loss(mode='ce')
metric = Metrics(num_classes=9, ignore_label=-1, device="cuda")

# optim and scheduler init
model_optimizer = optim.AdamW(model.parameters(), lr=opt.lr, eps=1e-6, weight_decay=1e-4)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

# train model
print("-----------------train-----------------")
min_loss = 10000
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()
    epoch_iou = utils.AverageMeter()
    epoch_f1 = utils.AverageMeter()
    epoch_acc = utils.AverageMeter()
    train_dataloader.sampler.set_epoch(epoch)
    with tqdm(total=(length - length % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for record in train_dataloader:

            inputs, labels = record
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            model_optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out, labels)

            loss.backward()
            # utils.clip_gradient(model_optimizer, 5)

            model_optimizer.step()

            metric.update(out, labels)
            _, iou = metric.compute_iou()
            _, f1 = metric.compute_f1()
            _, acc = metric.compute_pixel_acc()

            epoch_losses.update(loss.item(), opt.batch_size)
            epoch_iou.update(iou, opt.batch_size)
            epoch_f1.update(f1, opt.batch_size)
            epoch_acc.update(acc, opt.batch_size)

            t.set_postfix(
                loss='{:.6f}'.format(epoch_losses.avg),
                iou='{:.2f}'.format(epoch_iou.avg),
                f1='{:.2f}'.format(epoch_f1.avg),
                acc='{:.2f}'.format(epoch_acc.avg),
            )
            t.update(opt.batch_size)

    model_scheduler.step()
    if dist.get_rank() == 0 and loss.item()<min_loss:
        torch.save(model.state_dict(), "./Best_PVTv2_SegFormer.pth")

if dist.get_rank() == 0:
    torch.save(model.state_dict(), "./Last_PVTv2_SegFormer.pth")