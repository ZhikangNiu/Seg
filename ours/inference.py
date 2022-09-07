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

from ours.models import PVTv2_SegFormer


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
    x = np.argmax(mask, axis=1)+1
    label_codes = np.array(labels)
    x = label_codes[x.astype(np.uint8)]
    return x

@torch.no_grad()
def generate_test():

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = PVTv2_SegFormer('B1', 9).cuda()
    state_dict = torch.load("./Best_PVTv2_SegFormer.pth")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    labels = [0, 100, 200, 300, 400, 500, 600, 700, 800]

    dataset = SegData(root="/home/public/datasets/InternationalRaceTrackDataset/chusai_release/",split="test")

    ################### debug ####################
    img =cv2.imread("/home/public/datasets/InternationalRaceTrackDataset/chusai_release/testA/images/2.tif")
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.24138375, 0.25477552, 0.29299292],
                             [0.09506353, 0.09248942, 0.09274331]),
    ])
    x = transform_image(img)
    output = model(x.unsqueeze(0).cuda())
    pred = torch.softmax(output, dim=1).cpu().detach().numpy()
    x = np.argmax(pred, axis=1)
    label_codes = np.array(labels)
    x = label_codes[x.astype(np.uint8)]
    print(output)
    print(output.shape)
    print(output.max())
    print(output.min())
    print(output.mean())
    print(x)
    print(x.max())
    print(x.min())
    print("-----------------")
    label = cv2.imread("/home/public/datasets/InternationalRaceTrackDataset/chusai_release/train/labels/1.png",0)
    print(label.shape)
    print(label.max())
    print(label)
    #################### debug ####################

    # test_loader = dataloader.DataLoader(dataset=dataset, pin_memory=True)
    # print("-----------------test-----------------")
    # for image ,name in test_loader:
    #     image = image.cuda()
    #     output = model(image)
    #     pred = torch.softmax(output, dim=1).cpu().detach().numpy()
    #     # pred -> [1, 9, 512, 512]
    #     pred = semantic2mask(pred, labels=labels).squeeze().astype(np.uint16)
    #     cv2.imwrite(os.path.join(output_dir, name[0].replace("tif","png")), pred)



if __name__ == "__main__":
    generate_test()
    exit(0)

