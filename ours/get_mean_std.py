# -*- coding: utf-8 -*-
# @Time    : 2022-09-06 10:17
# @Author  : Zhikang Niu
# @FileName: get_mean_std.py
# @Software: PyCharm
import torch

from ours.datasets import SegData


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = SegData("/home/public/datasets/InternationalRaceTrackDataset/chusai_release/", split='train', transform=None)
    print(getStat(train_dataset))

"""
[0.24138375, 0.25477552, 0.29299292], [0.09506353, 0.09248942, 0.09274331]
"""