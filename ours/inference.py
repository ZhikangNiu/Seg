# -*- coding: utf-8 -*-
# @Time    : 2022-09-06 10:31
# @Author  : Zhikang Niu
# @FileName: inference.py
# @Software: PyCharm
import os
from collections import OrderedDict
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from datasets import SegData
import cv2
import numpy as np
from torch.utils.data import dataloader
from PIL import Image
from ours.models import PVTv2_Lawin
from tqdm import tqdm
import ttach as tta



def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
    if from_save_folder:
        save_path = conf.save_path
    else:
        save_path = conf.model_path

    state_dict = torch.load(save_path / 'model_{}'.format(fixed_str))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    self.model.load_state_dict(new_state_dict)
    if not model_only:
        self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
        self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

def semantic2mask(mask,labels):
    # 语义图转标签图  labels代表所有的标签 [0, 100, 200, 300, 400, 500, 600, 700, 800]
    x = np.argmax(mask, axis=1)
    label_codes = np.array(labels)
    x = label_codes[x.astype(np.uint8)]
    return x


@torch.no_grad()
def generate_test():

    output_dir = "./results/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    model = PVTv2_Lawin('B1', 9).cuda("cuda:1")
    state_dict = torch.load("./Best_PVTv2_B1_Lawin_bs32_8_1.pth")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    dataset = SegData(root="/home/public/datasets/InternationalRaceTrackDataset/chusai_release/",split="test")
    test_loader = dataloader.DataLoader(dataset=dataset, pin_memory=True)
    print("-----------------test-----------------")
    for image ,name in tqdm(test_loader):
        image = image.cuda("cuda:1")
        output = model(image)

        pred = torch.softmax(output, dim=1).cpu().detach().numpy()
        # pred -> [1, 9, 512, 512]
        pred = semantic2mask(pred, labels=labels).squeeze().astype(np.uint8)
        Image.fromarray(pred).save(os.path.join(output_dir, name[0]).replace("tif","png"))
        #cv2.imwrite(os.path.join(output_dir, name[0].replace("tif","png")), pred)


if __name__ == "__main__":
    generate_test()
    exit(0)

