#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
UGVKP Dataloader
jieli_cn@163.com
"""

import os
import imageio
import cv2
import numpy as np
import PIL

import torch
from training.datasets.coco_data.heatmap import putGaussianMaps
from training.datasets.ugv_data.ImageAugmentation import (aug_croppad, aug_flip, aug_rotate, aug_scale)
from training.datasets.ugv_data.paf import putVecMaps
from training.datasets.coco_data.preprocessing import (inception_preprocess, rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

import UGV_config as CF

# ----- For debug -----
from PIL import Image, ImageDraw
from network.utility import show_img


class UGVKeyPoints(Dataset):
    def __init__(self, root, index_list, data, feat_stride, preprocess='rtpose', transform=None,
                 target_transform=None, params_transform=None, numkeypoints=4, numlims=4, aug=False, classification = False):
        print("UGV keypoints: init data loader")
        self.params_transform = params_transform

        # 网络输入的图片大小
        self.params_transform['crop_size_x'] = CF.IMAGE_W
        self.params_transform['crop_size_y'] = CF.IMAGE_H

        self.params_transform['stride'] = feat_stride

        # add preprocessing as a choice, so we don't modify it manually.
        self.preprocess = preprocess
        self.data = data

        self.root = root
        # self.mask_dir = mask_dir
        self.numSample = len(index_list)

        self.index_list = index_list

        self.transform = transform
        self.target_transform = target_transform

        self.numkeypoints = numkeypoints  # Mod by Jie
        self.numlims = numlims

        self.classification = classification   #
        self.aug = aug
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1)
        ])
        _msg = 'ON' if self.aug else 'OFF'
        print('Online Data Augmentation: {}'.format(_msg))
        print('Input image process:', preprocess)

    def get_anno(self, meta_data):
        """
        get meta information
        """
        anno = dict()
        anno['dataset'] = meta_data['dataset']

        anno['factor'] = meta_data['img_height']/CF.IMAGE_H
        # anno['factor_h'] = meta_data['img_height']/CF.IMAGE_H
        # anno['factor_w'] = meta_data['img_width']/CF.IMAGE_W
        # anno['factor_h'] == anno['factor_w']

        # anno['img_height'] = int(meta_data['img_height'])
        # anno['img_width'] = int(meta_data['img_width'])
        anno['img_height'] = CF.IMAGE_H
        anno['img_width'] = CF.IMAGE_W

        anno['isValidation'] = meta_data['isValidation']

        anno['objpos'] = np.array(meta_data['objpos'])

        anno['scale_provided'] = meta_data['scale_provided']
        anno['scale_provided_other'] = meta_data['scale_provided_other']

        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])

        # ---
        anno['joint_others'] = np.array(meta_data['joint_others'])
        anno['objpos_other'] = np.array(meta_data['objpos_other'])

        return anno

    def _process_boundary_points(self, point):
        # x, y = point
        # print(x, y)
        if point[0] == 0:
            point[0] = 1
        if point[1] == 0:
            point[1] = 1
        if point[0] == CF.IMAGE_W:
            point[0] = CF.IMAGE_W - 1
        if point[1] == CF.IMAGE_H:
            point[1] = CF.IMAGE_H - 1

    def remove_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        mask = np.logical_or.reduce((meta['joint_self'][:, 0] >= crop_x,
                                     meta['joint_self'][:, 0] < 0,
                                     meta['joint_self'][:, 1] >= crop_y,
                                     meta['joint_self'][:, 1] < 0))
        # out_bound = np.nonzero(mask)
        # print(mask.shape)
        meta['joint_self'][mask == True, :] = (1, 1, 2)
        if (meta['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((meta['joint_others'][:, :, 0] >= crop_x,
                                         meta['joint_others'][:, :, 0] < 0,
                                         meta['joint_others'][:, :, 1] >= crop_y,
                                         meta['joint_others'][:, :, 1] < 0))
            meta['joint_others'][mask == True, :] = (1, 1, 2)

        return meta

    def process_boundary_points(self, keypoints_matrix):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        x_0 = keypoints_matrix[0, 0]
        y_0 = keypoints_matrix[0, 1]
        x_1 = keypoints_matrix[1, 0]
        y_1 = keypoints_matrix[1, 1]
        x_2 = keypoints_matrix[2, 0]
        y_2 = keypoints_matrix[2, 1]
        x_3 = keypoints_matrix[3, 0]
        y_3 = keypoints_matrix[3, 1]

        if x_0 < 0:
            keypoints_matrix[0, 0] = 5  # set new keypoint x = 5
            keypoints_matrix[0, 1] = y_1 + (y_0 - y_1) / (x_0 - x_1) * (5 - x_1)

        if x_3 < 0:
            keypoints_matrix[3, 0] = 5
            keypoints_matrix[3, 1] = y_2+ (y_3 - y_2) / (x_3 - x_2) * (5 - x_2)

            # keypoint 1 or 2 outside x= crop_x
        if x_1 >= crop_x:
            keypoints_matrix[1, 0] = crop_x - 5
            keypoints_matrix[1, 1] = y_0 + (y_1 - y_0) / (x_1 - x_0) * (crop_x - 5 - x_0)

        if x_2 >= crop_x:
            keypoints_matrix[2, 0] = crop_x - 5
            keypoints_matrix[2, 1] = y_3 + (y_2 - y_3) / (x_2 - x_3) * (crop_x - 5 - x_3)

            # keypoint 0 or 3 outside y= 0
        if y_0 < 0:
            keypoints_matrix[0, 1] = 5  # set new keypoint y = 5
            keypoints_matrix[0, 0] = x_1 + (x_0 - x_1) / (y_0 - y_1) * (5 - y_1)

        if y_3 < 0:
            keypoints_matrix[3, 1] = 5
            if y_3 == y_2:
                keypoints_matrix[3, 0] = x_0
            else:
                keypoints_matrix[3, 0] = x_2 + (x_3 - x_2) / (y_3 - y_2) * (5 - y_2)

            # keypoint 0 or 3 outside y= crop_x
        if y_0 >= crop_y:
            keypoints_matrix[0, 1] = crop_y - 5
            if y_0 == y_1:
                keypoints_matrix[0, 0] = x_3
            else:
                keypoints_matrix[0, 0] = x_1 + (x_0 - x_1) / (y_0 - y_1) * (crop_y - 5 - y_1)

        if y_3 >= crop_y:
            keypoints_matrix[3, 1] = crop_y - 5
            keypoints_matrix[3, 0] = x_2 + (x_3 - x_2) / (y_3 - y_2) * (crop_x - 5 - y_2)

            # keypoint 1 or 2 outside y= 0
        if y_1 < 0:
            keypoints_matrix[1, 1] = 5  # set new keypoint y = 5
            keypoints_matrix[1, 0] = x_0 + (x_1 - x_0) / (y_1 - y_0) * (5 - y_0)

        if y_2 < 0:
            keypoints_matrix[2, 1] = 5
            if y_2 == y_3:
                keypoints_matrix[2, 0] = x_1
            else:
                keypoints_matrix[2, 0] = x_3 + (x_2 - x_3) / (y_2 - y_3) * (5 - y_3)

            # keypoint 1 or 2 outside y= crop_x
        if y_1 >= crop_y:
            keypoints_matrix[1, 1] = crop_y - 5
            if y_0 == y_1:
                keypoints_matrix[1, 0] = x_2
            else:
                keypoints_matrix[1, 0] = x_0 + (x_1 - x_0) / (y_1 - y_0) * (crop_y - 5 - y_0)

        if y_2 >= crop_y:
            keypoints_matrix[2, 1] = crop_y - 5
            keypoints_matrix[2, 0] = x_3 + (x_2 - x_3) / (y_2 - y_3) * (crop_x - 5 - y_3)
        return keypoints_matrix

    # ---- from Chongyu, filter out illegal joint after the augmentation
    def process_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        # for i in range(meta['joint_self'].shape[0]):
        # keypoint 0 or 3 outside x= 0
        meta['joint_self'][:, :2] = self.process_boundary_points(meta['joint_self'][:, :2])
        if (meta['numOtherPeople'] != 0):
            for i in range(meta['numOtherPeople']):
                meta['joint_others'][i, :, :2] = self.process_boundary_points(meta['joint_others'][i, :, :2])
        return meta

    def get_ground_truth(self, meta, mask_miss, numkeypoints=4, numlims=4):  # Mod by Jie

        stride = self.params_transform['stride']

        mode = self.params_transform['mode']
        crop_size_y = self.params_transform['crop_size_y']
        crop_size_x = self.params_transform['crop_size_x']
        num_parts = self.params_transform['np']
        nop = meta['numOtherPeople']
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        channels = (num_parts + 1) * 2
        heatmaps = np.zeros((int(grid_y), int(grid_x), numkeypoints+1))  # Mod by Jie. channels: background + numkeypoints
        pafs = np.zeros((int(grid_y), int(grid_x), numlims*2))           # Mod by Jie

        mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 / stride,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mask_miss = mask_miss / 255.
        mask_miss = np.expand_dims(mask_miss, axis=2)  # (h, w) ---> (h, w, 1)

        heat_mask = np.repeat(mask_miss, numkeypoints+1, axis=2)    # Mod by Jie
        paf_mask = np.repeat(mask_miss, numlims*2, axis=2)          # Mod by Jie

        # confidance maps for body parts
        for i in range(numkeypoints):
            if (meta['joint_self'][i, 2] <= 1):
                center = meta['joint_self'][i, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(center, gaussian_map, params_transform=self.params_transform)
            for j in range(nop):
                if (meta['joint_others'][j, i, 2] <= 1):
                    center = meta['joint_others'][j, i, :2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(center, gaussian_map, params_transform=self.params_transform)

        # pafs
        mid_1 = [0, 1, 2, 3]
        mid_2 = [1, 2, 3, 0]

        thre = 1
        for i in range(numlims):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            if (meta['joint_self'][mid_1[i], 2] <= 1 and meta['joint_self'][mid_2[i], 2] <= 1):
                # if语句 判断kepoint点是否有效，对应的label为：1有效，2无效
                centerA = meta['joint_self'][mid_1[i], :2]
                centerB = meta['joint_self'][mid_2[i], :2]
                # self._process_boundary_points(centerA)
                # self._process_boundary_points(centerB)

                vec_map = pafs[:, :, 2 * i:2 * i + 2]
                #                    print vec_map.shape
                pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                centerB=centerB,
                                                                accumulate_vec_map=vec_map,
                                                                count=count, params_transform=self.params_transform)
            for j in range(nop):
                if (meta['joint_others'][j, mid_1[i], 2] <= 1 and meta['joint_others'][j, mid_2[i], 2] <= 1):
                    centerA = meta['joint_others'][j, mid_1[i], :2]
                    centerB = meta['joint_others'][j, mid_2[i], :2]
                    vec_map = pafs[:, :, 2 * i:2 * i + 2]
                    pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                    centerB=centerB,
                                                                    accumulate_vec_map=vec_map,
                                                                    count=count, params_transform=self.params_transform)
        # background
        heatmaps[:, :, - 1] = np.maximum(1 - np.max(heatmaps[:, :, :numlims], axis=2), 0.)

        class_map = np.argmax(heatmaps, axis=2)

        id = np.where(mask_miss == 0)  # ignore mask
        # print(mask_miss.shape)
        class_map[id[0], id[1]] = 255  # ignore index for CrossEntropyLoss

        return heat_mask, heatmaps, paf_mask, pafs, class_map

    # def show_img(self, meta, img, filename, text='GT'):
    #     im = PIL.Image.fromarray(img)
    #     draw = ImageDraw.Draw(im)
    #     draw.text((10, 10), '{}'.format(text))
    #     num_keypoints = meta['joint_self'].shape[0]
    #     print('image size = {} * {}'.format(img.shape[0], img.shape[1]))
    #     for i in range(num_keypoints):
    #         p = meta['joint_self'][i, :2]
    #         draw.ellipse((p[0]-5,p[1]-5,p[0]+5, p[1]+5), 'white')
    #         draw.text((p[0] + 10, p[1]-10), '{}'.format(i))
    #     if (meta['numOtherPeople'] != 0):
    #         for i in range(meta['numOtherPeople']):
    #             for j in range(4):
    #                 p = meta['joint_others'][i, j, :2]
    #                 draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), 'white')
    #                 draw.text((p[0] + 10, p[1] - 10), '{}-{}'.format(i, j))
    #
    #     # im.show()
    #     im.save((filename + ".thumbnail", "JPEG"))

    def __getitem__(self, index):
        idx = self.index_list[index]

        # img_path = os.path.join(self.root, self.data[idx]['img_paths'])
        img_path = os.path.join(self.root, self.data[idx]['img_dir'], self.data[idx]['img_paths'])
        img = cv2.imread(img_path)
        # img = PIL.Image.open(os.path.join(self.root, self.data[idx]['img_paths']))
        # img = imageio.imread(os.path.join(self.root, self.data[idx]['img_paths']))
        # print('img path: ', os.path.join(self.root, self.data[idx]['img_paths']))

        # ---- Read mask image
        if "KIK" in self.data[idx]['dataset']:
            img_name = self.data[idx]['img_paths'][:-3] + 'png'
            # mask_img_path = os.path.join(self.mask_dir, img_name)
            mask_img_path = os.path.join(self.root, self.data[idx]['img_dir'], img_name)
            if os.path.isfile(mask_img_path):
                # mask_miss = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                mask_miss = imageio.imread(mask_img_path).astype(np.uint8)
            else:
                print('No mask image found: {}'.format(mask_img_path))
                mask_miss = (np.ones((img.shape[0], img.shape[1]), dtype=np.uint8))*255

        # robot dataset, no mask
        mask_miss = (np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)) * 255
        # ---- end of Read mask image
        # mask_miss = (np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)) * 255
        # print(self.root + 'mask2014/val2014_mask_miss_' + img_idx + 'png')

        # Get infor from the loaded json
        meta_data = self.get_anno(self.data[idx])

        if self.aug:
            img = PIL.Image.fromarray(img)
            img = self.aug_transform(img)  # Randomly change the brightness, contrast and saturation of an image.
            img = np.asarray(img)

            # 缩放，缩放需配合aug_croppad，才能保证输出大小一致
            meta_data, img, mask_miss = aug_scale(meta_data, img, mask_miss, self.params_transform)
            # print('aug_scale', img.shape)
            #
            meta_data, img, mask_miss = aug_rotate(meta_data, img, mask_miss, self.params_transform)    # 旋转
            # print('aug_rotate', img.shape)
            #
            meta_data, img, mask_miss = aug_croppad(meta_data, img, mask_miss, self.params_transform)   # 左右平移和crop
            # print('aug_croppad', img.shape)
            #
            # meta_data, img, mask_miss = aug_flip(meta_data, img, mask_miss, self.params_transform)      # 左右翻转
            # print('aug_flip', img.shape)
            # TODO: change the types of the key points after flip
            meta_data = self.process_illegal_joint(meta_data)

        meta_data = self.remove_illegal_joint(meta_data)
        # from Chongyu, filter out illegal joint after the augmentation
        # meta_data = self.process_illegal_joint(meta_data)

        heat_mask, heatmaps, paf_mask, pafs, class_map = self.get_ground_truth(meta_data, mask_miss,
                                                                    numkeypoints=self.numkeypoints,
                                                                    numlims=self.numlims)

        # --- Check classification GT
        # img_name = os.path.join('./viz_GT', self.data[idx]['img_paths'])
        # h, w = class_map.shape
        # class_gt = np.zeros((h, w, 3), dtype=np.uint8)
        #
        # id0 = np.where(class_map == 0)  # kp0
        # class_gt[id0[0], id0[1], :] = 255, 0, 0  # R
        #
        # id1 = np.where(class_map == 1)  # kp1
        # class_gt[id1[0], id1[1], :] = 0, 255, 0  # G
        #
        # id2 = np.where(class_map == 2)  # kp2
        # class_gt[id2[0], id2[1], :] = 0, 0, 255  # B
        #
        # id3 = np.where(class_map == 3)  # kp3
        # class_gt[id3[0], id3[1], :] = 0, 0, 0   # black
        #
        # id4 = np.where(class_map == 4)  # background
        # class_gt[id4[0], id4[1], :] = 255, 255, 255  # white
        #
        # id = np.where(class_map == 255)  # ignore
        # class_gt[id[0], id[1], :] = 125, 125, 125  # white
        #
        # FACTOR = 4
        # im = Image.fromarray(class_gt)
        # im = im.resize((int(w * FACTOR), int(h * FACTOR)), resample=PIL.Image.ANTIALIAS)
        # im.save(img_name)
        # # imageio.imwrite(img_name, class_gt)
        # --- end: Check classification GT

        # show_img(meta_data, img, os.path.join('./viz_GT', self.data[idx]['img_paths']), text='GT')

        # image preprocessing, which comply the model
        # trianed on Imagenet dataset
        if self.preprocess == 'rtpose':
            img = rtpose_preprocess(img)

        elif self.preprocess == 'vgg':
            img = vgg_preprocess(img)

        elif self.preprocess == 'inception':
            img = inception_preprocess(img)

        elif self.preprocess == 'ssd':
            img = ssd_preprocess(img)

        img = torch.from_numpy(img)
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)).astype(np.float32))  # channel last --> channel first
        heat_mask = torch.from_numpy(heat_mask.transpose((2, 0, 1)).astype(np.float32))
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))
        paf_mask = torch.from_numpy(paf_mask.transpose((2, 0, 1)).astype(np.float32))

        if self.classification:
            return img, class_map, heat_mask, pafs, paf_mask, img_path
        else:
            return img, heatmaps, heat_mask, pafs, paf_mask, img_path

    def __len__(self):
        return self.numSample


if __name__ == '__main__':
    from training.datasets.kiktech import get_loader
    import kik_config as CF

    json_path = '/home/jie/kikprogram/robot/dataset/robot_keypoints_20190426_640x360.json'
    preprocess = 'rtpose'

    train_loader = get_loader(json_path, '', '', None, 8, preprocess, batch_size=1,
                            params_transform = CF.params_transform,
                            shuffle=False,
                            training=True,
                            num_workers=0,
                            # aug=True
                            # aug=False,
                              )

    viz_GT = True
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask, img_paths) in enumerate(train_loader):

        print(i, img.shape, heatmap_target.shape)