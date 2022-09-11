# -*- coding: utf-8 -*-
# @Time    : 2022-09-08 9:28
# @Author  : Zhikang Niu
# @FileName: visual_results.py
# @Software: PyCharm

import os
import numpy as np
import cv2

COLOR = np.array([
    [0, 0, 0],
    [0,255,255],
    [255,255, 0],
    [0, 255, 0],
    [0,0,255],
    [254, 0, 0],
    [255, 0, 255],
    [100, 100, 0],
    [255, 255, 255],])


def dye_single_img(img,file_path,output_dir):
    single_img_path = os.path.join(file_path,img)
    label_img = cv2.imread(single_img_path,0)
    dye_img = COLOR[label_img]
    cv2.imwrite(os.path.join(output_dir, img), dye_img)


def main(file_path,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_img_files = os.listdir(file_path)

    for img in all_img_files:
        dye_single_img(img,file_path,output_dir)



if __name__ == '__main__':
    # file_path = "/home/public/datasets/InternationalRaceTrackDataset/chusai_release/train/labels"
    # output_dir = "./dye_train_img_results/"
    # main(file_path,output_dir)
    res = cv2.imread("res.png",0)
    print(res)
    res_canny = cv2.imread("/home/public/datasets/InternationalRaceTrackDataset/chusai_release/train/labels/2.png",0)
    print(res_canny)
    d = np.argwhere(res != res_canny)
    print(len(d))
    print(512*512)