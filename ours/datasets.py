# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 20:01
# @Author  : Zhikang Niu
# @FileName: datasets.py
# @Software: PyCharm
import os

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple


class SegData(Dataset):
    CLASSES = [
        'water',
        'building',
        'grassland',
        'transportation',
        'woodland',
        'bare-earth',
        'cultivated-land',
        'other',
        'background',
    ]

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'test']
        self.split = 'testA' if split == 'test' else 'train'
        self.transform = transform
        if self.transform is None:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.24138375, 0.25477552, 0.29299292],
                                     [0.09506353, 0.09248942, 0.09274331]),
            ])
            # self.transform_label = transforms.Compose([
            #     transforms.ToTensor(),
            # ])
        else:
            self.transform_image = transform
            self.transform_label = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / self.split / 'images'
        self.img_files = list(img_path.glob('*.tif'))

        print(f"Found {len(self.img_files)} {self.split} images.")


    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.split == 'train':
            img_path = str(self.img_files[index])
            lbl_path = img_path.replace('images', 'labels').replace('.tif', '.png')
            image = cv2.imread(img_path)
            label = Image.open(lbl_path)

            if self.transform is None:
                image = self.transform_image(image)
                label = np.array(label) / 100
                label = torch.tensor(label)
            else:
                image = Image.fromarray(image)
                image = self.transform_image(image)
                label = self.transform_label(label)
                pro_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.24138375, 0.25477552, 0.29299292],
                                         [0.09506353, 0.09248942, 0.09274331])
                ])

                image = pro_transform(image)
                label = np.array(label) / 100
                label = torch.tensor(label)

            return image, label.squeeze().long()
        else:
            img_path = str(self.img_files[index])
            image = cv2.imread(img_path)
            name = img_path[80:]
            image = self.transform_image(image)
            return image, name

if __name__ == '__main__':
    dataset = SegData(root='/home/public/datasets/InternationalRaceTrackDataset/chusai_release/', split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (image, label) in enumerate(dataloader):
        print(image.shape)
        print(label.shape)
        print(label)
        print(label.max())
        print(label.min())

