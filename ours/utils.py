# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 19:58
# @Author  : Zhikang Niu
# @FileName: utils.py
# @Software: PyCharm
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count