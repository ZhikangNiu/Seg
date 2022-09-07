# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 19:58
# @Author  : Zhikang Niu
# @FileName: config.py
# @Software: PyCharm
import argparse


def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size, default=64')
    parser.add_argument('--data_root', type=str, default='/home/public/datasets/InternationalRaceTrackDataset/chusai_release/', help='...')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for, default=10')
    parser.add_argument('--resume', action='store_true', default=False
                        , help='...')
    parser.add_argument('--pretrained',action='store_true', default=False
                        , help='...')
    parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=3e-5')
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument("--local_rank", type=int,default=-1, help='local rank for DistributedDataParallel')
    opt = parser.parse_args()

    return opt