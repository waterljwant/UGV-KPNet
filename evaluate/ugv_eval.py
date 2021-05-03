#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import time
# import datetime
from shutil import copyfile



import cv2
import numpy as np
import json
# import pandas as pd
from pycocotools.coco import COCO
from evaluate.tool_ugv_eval import UGV_Eval

from network.utility import viz_featuremaps

import torch
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network.ugv_post import decode_pose
from network import im_transform


import UGV_config as CF

# NUM_KEYPOINTS = 4
ORDER_KIK = [0, 1, 2, 3]
#
'''
images marked by 'isValidation = 1', as our testing dataset.
'''
FLAG_VAL = 1.0  # validation data
# FLAG_VAL = 0.0  # training data

# hyperparameters
TH_iou = 0.8  # The predicted pallet is a right hit if its IoU with GT is larger than TH_iou.


def get_anno(meta_data):
    """
    get meta information
    """
    anno = dict()
    anno['dataset'] = meta_data['dataset']
    anno['img_height'] = int(meta_data['img_height'])
    anno['img_width'] = int(meta_data['img_width'])

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


# 这个evaludtion 没有考虑卡板被重复检测的情况
def _eval_kik_Eud(outputs, json_data, FLAG_VAL):
    idx_val = -1
    hit_count_img = 0       # number of images which have been detected successfully.
    img_count = 0           # number of images in the validation data set
    hit_count_pallet = 0    # number of pallets which have been detected successfully.
    pallet_count = 0        # number of pallets in the validation data set
    iou_list = list()
    dist_list = list()
    angle_list = list()
    hit_img_list = list()
    img_height = int(json_data[0]['img_height'])
    img_width  = int(json_data[0]['img_width'])

    for idx_gt in range(len(json_data)):  # 遍历GT
        if json_data[idx_gt]['isValidation'] == FLAG_VAL:
            idx_val += 1
            img_count += 1
            pallet_count += len(json_data[idx_gt]['joint_self'])/4 + len(json_data[idx_gt]['joint_others'])
            gt_keypoints_arr = np.array(json_data[idx_gt]['joint_self'])[:, :-1].astype(np.int32)[np.newaxis, :]

            if len(json_data[idx_gt]['joint_others']) > 0:
                gt_other = np.array(json_data[idx_gt]['joint_others'])[:, :, :-1].astype(np.int32)
                gt_keypoints_arr = np.concatenate((gt_keypoints_arr, gt_other), axis=0)

            d = np.zeros(4, dtype=np.float32)
            for idx_pd, res in enumerate(outputs):  # 对于每一个GT都需要遍历outputs
                if idx_gt == res["json_id"]:
                # if idx_val == res["json_id"]:
                    if idx_gt not in hit_img_list:
                        hit_count_img += 1
                        hit_img_list.append(idx_gt)

                    # print(res["json_id"])
                    # ------ calc IoU of the predicted pallet with the Ground truth pallet
                    # ---prediction
                    pd_keypoints = np.array(res['keypoints'])[:, :-1].astype(np.int32)
                    mask_pd = np.zeros((img_height, img_width))
                    cv2.fillConvexPoly(mask_pd, pd_keypoints, 1)
                    mask_pd = mask_pd.astype(np.bool)
                    # ---ground truth
                    for ig in range(gt_keypoints_arr.shape[0]):
                        gt_keypoints = gt_keypoints_arr[ig, :, :]
                        mask_gt = np.zeros((img_height, img_width))
                        cv2.fillConvexPoly(mask_gt, gt_keypoints, 1)
                        mask_gt = mask_gt.astype(np.bool)
                        # intersection = np.logical_and(mask_pd, mask_gt)
                        # union = np.logical_or(mask_pd, mask_gt)
                        inter_num = np.array(np.where(np.logical_and(mask_pd, mask_gt))).size
                        union_num = np.array(np.where(np.logical_or(mask_pd, mask_gt))).size
                        iou = 1.0 * inter_num / union_num
                        if iou > TH_iou:
                            hit_count_pallet += 1
                            iou_list.append(iou)
                            d[0] = np.sqrt(np.sum(np.square(pd_keypoints[0] - gt_keypoints[0])))
                            d[1] = np.sqrt(np.sum(np.square(pd_keypoints[1] - gt_keypoints[1])))
                            d[2] = np.sqrt(np.sum(np.square(pd_keypoints[2] - gt_keypoints[2])))
                            d[3] = np.sqrt(np.sum(np.square(pd_keypoints[3] - gt_keypoints[3])))
                            dist_list.append(d)
                            # print(pd_keypoints[0] - gt_keypoints[0])
                            # --- Calc the angel between gt_keypoints and pd_keypoints
                            x = pd_keypoints[1] - pd_keypoints[0]
                            y = gt_keypoints[1] - gt_keypoints[0]
                            Lx = np.sqrt(x.dot(x))
                            Ly = np.sqrt(y.dot(y))
                            # cos_angle = x.dot(y) / (Lx * Ly)  # if x==y, cos_angle=1.0000000002 and arccos will be nan
                            cos_angle = 1 if np.all(x == y) else x.dot(y) / (Lx * Ly)
                            angle = np.arccos(cos_angle) * 360.0 / 2 / np.pi
                            angle_list.append(angle)
                            # if np.isnan(angle):
                            #     print('angle is nan. cos_angle:', cos_angle, x, y)
                            # print('Hit a pallet in the image:', json_data[idx_gt]['img_paths'])
                            # print('distance', d, 'angle', angle)
                        # print('IoU', iou)
    avg_dist = np.average(np.asarray(dist_list), axis=0)
    avg_angle = np.average(np.asarray(angle_list), axis=0)
    mIoU = 0 if len(iou_list) == 0 else np.average(np.array(iou_list))
    print('---- Evaluation results ----',
          '\nhit_count_img: \t{}'.format(hit_count_img),
          '\nimg_count:     \t{}'.format(img_count),
          '\nimg hit rate:  \t{:.2f}%'.format(100.0*hit_count_img/img_count),
          '\nhit_count_pallet:\t{}'.format(hit_count_pallet),
          '\npallet_count:    \t{}'.format(pallet_count),
          '\npallet hit rate: \t{:.2f}%'.format(100.0 * hit_count_pallet / pallet_count),
          '\naverage dist: \t{}'.format(np.around(avg_dist, decimals=2)),
          '\naverage angle:\t{:.2f}'.format(avg_angle),
          '\nmIoU:\t{:.2f}%'.format(100.0*mIoU))
    return mIoU


def eval_kik_Eud(outputs, json_data):
    idx_val = -1
    hit_count_img = 0       # number of images which have been detected successfully.
    img_count = 0           # number of images in the validation data set
    hit_count_pallet = 0    # number of pallets which have been detected successfully.
    gt_pallet_count = 0        # number of pallets in the validation data set
    iou_list = list()
    dist_list = list()
    angle_list = list()
    hit_img_list = list()
    img_height = int(json_data[0]['img_height'])
    img_width  = int(json_data[0]['img_width'])

    for idx_gt in range(len(json_data)):  # 遍历GT
        if json_data[idx_gt]['isValidation'] == FLAG_VAL:
            gt_pallet_count += len(json_data[idx_gt]['joint_self'])/4 + len(json_data[idx_gt]['joint_others'])

    gt_dt = np.zeros(int(gt_pallet_count))
    pred_dt = np.zeros(len(outputs))
    print('Number of GT pallets:', gt_pallet_count, 'Number of detected pallets:', len(outputs))

    pallet_count_idx = 0
    for idx_gt in range(len(json_data)):  # 遍历GT
        if json_data[idx_gt]['isValidation'] == FLAG_VAL:
            idx_val += 1
            img_count += 1

            if img_count % 10 == 0:
                print("Evaluated {} images".format(img_count))

            # pallet_count_idx += len(json_data[idx_gt]['joint_self'])/4 + len(json_data[idx_gt]['joint_others'])
            gt_keypoints_arr = np.array(json_data[idx_gt]['joint_self'])[:, :-1].astype(np.int32)[np.newaxis, :]

            if len(json_data[idx_gt]['joint_others']) > 0:
                gt_other = np.array(json_data[idx_gt]['joint_others'])[:, :, :-1].astype(np.int32)
                gt_keypoints_arr = np.concatenate((gt_keypoints_arr, gt_other), axis=0)

            # --- Traversing the ground truth
            for ig in range(gt_keypoints_arr.shape[0]):
                gt_keypoints = gt_keypoints_arr[ig, :, :]
                mask_gt = np.zeros((img_height, img_width))
                cv2.fillConvexPoly(mask_gt, gt_keypoints, 1)
                mask_gt = mask_gt.astype(np.bool)

                iou_best = 0
                iou_idx = -1

                # --- find the highest IoU, record the IoU and index of the prediction.
                for idx_pd, res in enumerate(outputs):  # 对于每一个GT都需要遍历outputs
                    # 判断 GT的图片json ID和 predicts的图片json ID是否相同
                    if idx_gt == res["json_id"]:
                        # ---prediction
                        pd_keypoints = np.array(res['keypoints'])[:, :-1].astype(np.int32)
                        mask_pd = np.zeros((img_height, img_width))
                        cv2.fillConvexPoly(mask_pd, pd_keypoints, 1)
                        mask_pd = mask_pd.astype(np.bool)
                        # ------ calc IoU of the predicted pallet with the Ground truth pallet
                        inter_num = np.array(np.where(np.logical_and(mask_pd, mask_gt))).size
                        union_num = np.array(np.where(np.logical_or(mask_pd, mask_gt))).size
                        iou = 1.0 * inter_num / union_num
                        #
                        if iou > iou_best:
                            iou_best = iou
                            iou_idx = idx_pd

                # --- when the best prediction with highest iou is found, calc the evaluation metric.
                if iou_best > TH_iou and iou_idx > -1:
                    if pred_dt[iou_idx] == 1: # TODO check whether the pred_dt[iou_idx] is already set to 1
                        print('pred_dt[{}] is already set to 1'.format(iou_idx))
                    gt_dt[pallet_count_idx] = 1
                    pred_dt[iou_idx] = 1
                    # 1. update image hit count
                    if idx_gt not in hit_img_list:
                        hit_count_img += 1
                        hit_img_list.append(idx_gt)
                    # 2. calc metrics
                    hit_count_pallet += 1
                    iou_list.append(iou_best)
                    res_hit = outputs[iou_idx]
                    pd_keypoints = np.array(res_hit['keypoints'])[:, :-1].astype(np.int32)
                    # --- distance of each predicted keyppint to the gt point
                    d = np.zeros(4, dtype=np.float32)
                    d[0] = np.sqrt(np.sum(np.square(pd_keypoints[0] - gt_keypoints[0])))
                    d[1] = np.sqrt(np.sum(np.square(pd_keypoints[1] - gt_keypoints[1])))
                    d[2] = np.sqrt(np.sum(np.square(pd_keypoints[2] - gt_keypoints[2])))
                    d[3] = np.sqrt(np.sum(np.square(pd_keypoints[3] - gt_keypoints[3])))
                    dist_list.append(d)
                    # --- Calc the angel between gt_keypoints and pd_keypoints
                    x = pd_keypoints[1] - pd_keypoints[0]
                    y = gt_keypoints[1] - gt_keypoints[0]
                    Lx = np.sqrt(x.dot(x))
                    Ly = np.sqrt(y.dot(y))
                    # cos_angle = x.dot(y) / (Lx * Ly)  # if x==y, cos_angle=1.0000000002 and arccos will be nan
                    cos_angle = 1 if np.all(x == y) else x.dot(y) / (Lx * Ly)
                    angle = np.arccos(cos_angle) * 360.0 / 2 / np.pi
                    angle_list.append(angle)

                # --- update pallet_count_idx
                pallet_count_idx += 1
            # --- End: Traversing the ground truth

    # TP
    TP = np.count_nonzero(gt_dt==1)  # np.sum(gt_dt)
    FP = np.count_nonzero(pred_dt==0)
    FN = np.count_nonzero(gt_dt==0)
    p = 1.0 * TP / (TP + FP) if (TP + FP) > 0 else 0
    r = 1.0 * TP / (TP + FN) if (TP + FN) > 0 else 0
    print('TP:{} FP:{} FN:{}'.format(TP, FP, FN))
    avg_dist = np.average(np.asarray(dist_list), axis=0)
    avg_angle = np.average(np.asarray(angle_list), axis=0)
    mIoU = 0 if len(iou_list) == 0 else np.average(np.array(iou_list))
    print('---- Evaluation results ----',
          '\nhit_count_img: \t{}'.format(hit_count_img),
          '\nimg_count:     \t{}'.format(img_count),
          '\nimg hit rate:  \t{:.2f}%'.format(100.0*hit_count_img/img_count),
          '\nhit_count_robot:\t{}'.format(hit_count_pallet),
          '\nrobot_count:    \t{}'.format(gt_pallet_count),
          '\nrobot hit rate: \t{:.2f}%'.format(100.0 * hit_count_pallet / gt_pallet_count),
          '\nPrecision:\t{:.2f}%'.format(100.0 * p),
          '\nRecall:\t{:.2f}%'.format(100.0 * r),
          '\naverage dist: \t{}'.format(np.around(avg_dist, decimals=2)),
          '\naverage angle:\t{:.2f}'.format(avg_angle),
          '\nmIoU:\t{:.2f}%'.format(100.0*mIoU))
    return mIoU


def eval_kik_OKS(outputs, json_path, imgIds):
    """Evaluate images on Coco test set
    :param outputs: list of dictionaries, the models' processed outputs
    :param dataDir: string, path to the MSCOCO data directory
    :param imgIds: list, all the image ids in the validation set
    :returns : float, the mAP score
    """
    with open('results.json', 'w') as f:
        json.dump(outputs, f)  
    annType = 'keypoints'
    prefix = 'person_keypoints'

    # initialize COCO ground truth api
    dataType = 'val2014'
    # annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    cocoGt = COCO(json_path)  # load annotations
    cocoDt = cocoGt.loadRes('results.json')  # load model outputs

    # running evaluation
    kikEval = UGV_Eval(cocoGt, cocoDt, annType)  # TODO, modify this Class
    kikEval.params.imgIds = imgIds
    kikEval.evaluate()
    kikEval.accumulate()
    kikEval.summarize()
    os.remove('results.json')
    # return Average Precision
    return kikEval.stats[0]


def get_multiplier(img_width=CF.IMAGE_W, inp_size=CF.IMAGE_W, multisacle=False):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    # scale_search = [0.5, 1., 1.5, 2, 2.5]
    scale_search = [0.8, 1.] if multisacle else [1.]  # Mod by Jie
    return [x * inp_size / (1. * img_width) for x in scale_search]


def get_kik_val(json_path):
    """Reads MSCOCO validation informatio
    :param file_path: string, the path to the MSCOCO validation file
    :returns : list of image ids, list of image file paths, list of widths,
               list of heights
    """
    # val_coco = pd.read_csv(file_path, sep='\s+', header=None)
    # image_ids = list(val_coco[1])
    # file_paths = list(val_coco[2])
    # heights = list(val_coco[3])
    # widths = list(val_coco[4])
    # return image_ids, file_paths, heights, widths

    # read json file
    # with open(json_path) as data_file:
    #     data = json.load(data_file)

    json_data = list()
    if isinstance(json_path, list):  # 将多个root

        root = os.path.dirname(json_path[0])
        for json_i in json_path:
            print(json_i)
            with open(json_i) as data_file:
                json_data_i = json.load(data_file)
                json_data.extend(json_data_i)
        #         print(len(json_data_i), type(json_data_i))
        # print(len(json_data))

    elif isinstance(json_path, str):
        root = os.path.dirname(json_path)
        print(json_path)
        with open(json_path) as data_file:
            json_data = json.load(data_file)

    return json_data, root


def get_outputs(multiplier, img, model, preprocess, numkeypoints=4, numlims=4):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], numkeypoints+1))
    paf_avg = np.zeros((img.shape[0], img.shape[1], numlims*2))
    max_scale = multiplier[-1]
    max_size = max_scale * img.shape[0]
    # padding
    max_cropped, _, _ = im_transform.crop_with_factor(img, max_size, factor=8, is_ceil=True)
    batch_images = np.zeros((len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_croped, im_scale, real_shape = im_transform.crop_with_factor(
            img, inp_size, factor=8, is_ceil=True)

        if preprocess == 'rtpose':
            im_data = rtpose_preprocess(im_croped)

        elif preprocess == 'vgg':
            im_data = vgg_preprocess(im_croped)

        elif preprocess == 'inception':
            im_data = inception_preprocess(im_croped)

        elif preprocess == 'ssd':
            im_data = ssd_preprocess(im_croped)

        batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]

    heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)  # channel first to channel last
    pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = im_transform.crop_with_factor(img, inp_size, factor=8, is_ceil=True)
        heatmap = heatmaps[m, :int(im_cropped.shape[0] / 8), :int(im_cropped.shape[1] / 8), :]
        heatmap = cv2.resize(heatmap, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = pafs[m, :int(im_cropped.shape[0] / 8), :int(im_cropped.shape[1] / 8), :]
        paf = cv2.resize(paf, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        paf = paf[0:real_shape[0], 0:real_shape[1], :]
        paf = cv2.resize(paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    return paf_avg, heatmap_avg


def append_result(json_id, person_to_joint_assoc, joint_list, outputs):
    """Build the outputs to be evaluated
    :param image_id: int, the id of the current image
    :param person_to_joint_assoc: numpy array of joints associations
    :param joint_list: list, list of joints
    :param outputs: list of dictionaries with the following keys: image_id,
                    category_id, keypoints, score
    """

    for ridxPred in range(len(person_to_joint_assoc)):
        one_result = {"json_id": json_id, "category_id": 1, "keypoints": [], "score": 0}

        keypoints = np.zeros((CF.NUM_KEYPOINTS, 3))

        for part in range(CF.NUM_KEYPOINTS):
            ind = ORDER_KIK[part]  # TODO
            index = int(person_to_joint_assoc[ridxPred, ind])

            if -1 == index:
                keypoints[part, 0] = 0
                keypoints[part, 1] = 0
                keypoints[part, 2] = 0

            else:
                keypoints[part, 0] = joint_list[index, 0] + 0.5
                keypoints[part, 1] = joint_list[index, 1] + 0.5
                keypoints[part, 2] = 1

        one_result["score"] = person_to_joint_assoc[ridxPred, -2] * person_to_joint_assoc[ridxPred, -1]
        # one_result["keypoints"] = list(keypoints.reshape(int(NUM_KEYPOINTS*3)))
        one_result["keypoints"] = keypoints.tolist()  #

        # print('append in ', image_id)
        outputs.append(one_result)


def handle_paf_and_heat(normal_heat, flipped_heat, normal_paf, flipped_paf):
    """Compute the average of normal and flipped heatmap and paf
    :param normal_heat: numpy array, the normal heatmap
    :param normal_paf: numpy array, the normal paf
    :param flipped_heat: numpy array, the flipped heatmap
    :param flipped_paf: numpy array, the flipped  paf
    :returns: numpy arrays, the averaged paf and heatmap
    """

    '''
    # Method 2
    # The order to swap left and right of heatmap
    swap_heat = np.array((1, 0, 3, 2, 4))  # TODO
    swap_paf = np.array((0, 1, 6, 7, 4, 5, 2, 3))  # TODO

    flipped_paf = flipped_paf[:, ::-1, :]

    # The pafs are unit vectors, The x will change direction after flipped.
    # not easy to understand, you may try visualize it.
    flipped_paf[:, :, swap_paf[1::2]] = flipped_paf[:, :, swap_paf[1::2]]
    flipped_paf[:, :, swap_paf[::2]] = -flipped_paf[:, :, swap_paf[::2]]
    averaged_paf = (normal_paf + flipped_paf[:, :, swap_paf]) / 2.
    averaged_heatmap = (normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.
    '''

    # ---Method 1
    # ---- revise the type of the keypoints,  0<--> 1, 2 <--> 3
    flipped_heat = flipped_heat[:, ::-1, :]  # (480, 640, 5)
    # print(flipped_heat[10, 20, :])
    flipped_heat[:, :, [0, 1]] = flipped_heat[:, :, [1, 0]]
    flipped_heat[:, :, [2, 3]] = flipped_heat[:, :, [3, 2]]
    # print(flipped_heat[10, 20, :])
    averaged_heatmap = (normal_heat + flipped_heat) / 2.
    # averaged_heatmap = np.maximum(normal_heat, flipped_heat[:, ::-1, :])

    # ---- revise the direction of paf
    # flipped_paf = -flipped_paf[:, ::-1, :]  # x --> -x, y --> -y
    flipped_paf = flipped_paf[:, ::-1, :]
    # flipped_paf[:, :, 1] = -flipped_paf[:, :, 1]  # limb0, y
    # flipped_paf[:, :, 3] = -flipped_paf[:, :, 3]  # limb1, y
    # flipped_paf[:, :, 5] = -flipped_paf[:, :, 5]  # limb2, y
    # flipped_paf[:, :, 7] = -flipped_paf[:, :, 7]  # limb3, y
    flipped_paf[:, :, 1::2] = -flipped_paf[:, :, 1::2]  # y --> -y

    # print(flipped_paf[10, 20, :])
    flipped_paf[:, :, [2, 6]] = flipped_paf[:, :, [6, 2]]
    flipped_paf[:, :, [3, 7]] = flipped_paf[:, :, [7, 3]]
    # print(flipped_paf[10, 20, :])
    # print(normal_paf[10, 20, :], flipped_paf[10, 20, :])
    averaged_paf = (normal_paf + flipped_paf) / 2.  # Can not use np.maximum

    return averaged_paf, averaged_heatmap, flipped_paf, flipped_heat


from PIL import Image, ImageDraw, ImageFont
def show_img(meta, img, filename, text='GT'):
    # img = img[:, :, ::-1]
    # im = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("arial.ttf", 10)
    # font = ImageFont.truetype("/Users/jie/Library/Fonts/msyh.ttf", 13)
    # draw.text((10, 10), '{}'.format(text), font)
    draw.text((10, 10), '{}'.format(text))

    num_keypoints = len(meta['joint_self'])

    for i in range(num_keypoints):
        p = np.asarray(meta['joint_self'])[i, :2]
        draw.ellipse((p[0]-5,p[1]-5,p[0]+5, p[1]+5), 'white')
        draw.text((p[0] + 10, p[1]-10), '{}'.format(i))

    if meta['numOtherPeople'] > 0:
        # print(filename)
        for i in range(meta['numOtherPeople']):
            for j in range(4):
                p = np.asarray(meta['joint_others'])[i, j, :2]
                draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), 'white')
                draw.text((p[0] + 10, p[1] - 10), '{}-{}'.format(i, j))

    img.save(filename, "JPEG")


def show_GT(json_path, out_dir='./', flag=None):
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print('mkdir:', out_dir)
    print('save vis_GT images to:', out_dir)

    val_json, root = get_kik_val(json_path)

    print('Flag', flag)
    val_count = 0
    for i in range(len(val_json)):
        if flag == 'TRAIN' and val_json[i]['isValidation'] == 1.0:  # only train
            continue
        if flag == 'VAL' and val_json[i]['isValidation'] == 0.0:  # only val
            continue
        val_count += 1
    print("Total number of images {}".format(val_count))

    for json_id in range(len(val_json)):
        if flag == 'TRAIN' and val_json[json_id]['isValidation'] == 1.0:  # only train
            continue
        if flag == 'VAL' and val_json[json_id]['isValidation'] == 0.0:  # only val
            continue
        img_path = os.path.join(root, val_json[json_id]['img_dir'], val_json[json_id]['img_paths'])
        # oriImg = cv2.imread(img_path)  # img_paths[i], BGR

        oriImg = Image.open(img_path)  # RGB
        show_img(val_json[json_id], oriImg, os.path.join(out_dir, val_json[json_id]['img_paths']), text='GT')


# ----------------- Save results to TXT -----------------
def savekp2txt(count, results_list, gt_keypoints, image_path, joints, limbs, pallets):
    results_list.append(count)
    # --- image path
    results_list.append('Image:'+image_path)

    # gt_keypoints
    results_list.append('Ground truth - key points:')
    results_list.append(gt_keypoints)

    # --- joint_list:   (x, y, probability, keypoint_ID, keypoint_type)
    results_list.append('Candidate key points:')
    results_list.append(joints)

    # --- connected_limbs: list of limb,
    # 每个limb 对应一个数组: [joint_src_id, joint_dst_id, limb_score, joint_src_index, joint_dst_index]
    results_list.append('Connected limbs:')
    results_list.append(np.asarray(limbs))

    # --- pallet_to_joint_assoc:
    # 2d np.array of size num_pallet x (NUM_JOINTS+2). For each pallet found:
    # First NUM_JOINTS columns contain the index (in joint_list) of the joints associated
    # with that pallet (or -1 if their i-th joint wasn't found)
    # 2nd-to-last column: Overall score of the joints+limbs that belong to this pallet
    # Last column: Total count of joints found for this pallet
    results_list.append('Assembled pallets:')
    results_list.append(pallets)
    results_list.append('')
    return


def savep2txt(gt_keypoints, image_path, joints, limbs, pallets, root='./'):
    pallets_num = len(pallets)
    # print('pallets_num', pallets_num)
    kp = np.ones((pallets_num, 8)) * -1.
    for p in range(pallets_num):
        for i in range(4):
            idx = pallets[p, i]
            # print(idx)
            if idx >= 0:
                idx = idx.astype(np.int32)
                kp[p, i * 2] = joints[idx, 0]
                kp[p, i * 2 + 1] = joints[idx, 1]
            else:  # idx==-1.
                kp[p, i * 2] = -1
                kp[p, i * 2 + 1] = -1
            # print(kp)
    kp = kp.astype(np.int32)
    np.savetxt(os.path.join(root, image_path[:-4] + '_pd.txt'), kp, fmt='%d')

    gt = gt_keypoints.astype(np.int32)
    gt = gt.reshape((gt.shape[0], -1))
    np.savetxt(os.path.join(root, image_path[:-4] + '_gt.txt'), gt, fmt='%d')
    return


def run_eval(json_path, model, preprocess, vis_dir=None, flag=None):
    """Run the evaluation on the test set and report mAP score
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    # This txt file is fount in the caffe_rtpose repository:
    # https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master
    # img_ids, img_paths, img_heights, img_widths = get_kik_val(image_list_txt)

    _msg = 'validation' if FLAG_VAL else 'tarining'
    print("Processing Images in {} set".format(_msg))

    _msg = 'ON' if 'flip' in flag else 'OFF'
    b_flip = True if 'flip' in flag else False
    print('Flip: {}'.format(_msg))

    _msg = 'ON' if 'multi-scale' in flag else 'OFF'
    b_multiscale = True if 'multi-scale' in flag else False
    print('Multi-scale: {}'.format(_msg))

    _msg = 'ON' if 'solid' in flag else 'OFF'
    b_solid = True if 'solid' in flag else False
    print('Solid cycle: {}'.format(_msg))

    _msg = 'ON' if 'post-processing' in flag else 'OFF'
    b_postprocessing = True if 'post-processing' in flag else False
    print('Post-processing: {}'.format(_msg))


    # save_kp_2_txt = True
    save_kp_2_txt = False
    if save_kp_2_txt:
        results_list = list()

    val_json, root = get_kik_val(json_path)
    multiplier = get_multiplier(CF.IMAGE_W, CF.IMAGE_W, b_multiscale)

    val_count = 0
    for i in range(len(val_json)):
        if val_json[i]['isValidation'] == FLAG_VAL:
            val_count += 1
    print("Total number of images {}".format(val_count))

    tic = time.time()
    outputs = []
    val_count = 0
    # iterate all val images
    for json_id in range(len(val_json)):
    # for json_id in range(12):
        # ---- all
        # val_count += 1
        # ---- only FLAG_VAL
        if val_json[json_id]['isValidation'] == FLAG_VAL:
            val_count += 1
            # pass
        else:
            continue

        if val_count % 30 == 0 and val_count != 0:
            print("Processed {} images".format(val_count))

        # print('image path: ', os.path.join(image_dir, val_json[i]['img_paths']))
        # img_path = os.path.join(image_dir, val_json[json_id]['img_paths'])
        # root = os.path.dirname(json_path)
        img_name = val_json[json_id]['img_paths']
        img_path = os.path.join(root, val_json[json_id]['img_dir'], img_name)
        oriImg = cv2.imread(img_path)  # img_paths[i]
        # Get the shortest side of the image (either height or width)
        # shape_dst = np.min(oriImg.shape[0:2])

        # ------ flag，用于调试时查看特定的图片的输出
        # flag = None
        # flag = img_name if img_name == '20190226131815_37.jpg' else flag
        # flag = img_name if img_name == '20190226131815_40.jpg' else flag
        # flag = img_name if img_name == '20190226131815_42.jpg' else flag
        # flag = img_name if img_name == '20190226131815_59.jpg' else flag
        # if flag:
        #     print(flag)

        # choose which post-processing to use, our_post_processing
        # got slightly higher AP but is slow.
        # param = {'thre1': 0.2, 'thre2': 0.05, 'thre3': 0.5}
        param = {'thre1': 0.3, 'thre2': 0.05, 'thre3': 0.5}
        # thre1, heatmap
        # thre2, score_intermed_pts
        # thre3,
        viz = True if vis_dir is not None else False

        if b_flip:
            # ---- Step1: Get results of original image
            orig_paf, orig_heat = get_outputs(multiplier, oriImg, model, preprocess)
            # ---- Step2: Get results of flipped image, (480, 640, 3)
            swapped_img = oriImg[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, model, preprocess)
            # ---- Step3: compute averaged heatmap and paf
            # time1 = time.time()
            paf, heatmap, fpaf, fheat = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)
            # time2 = time.time()
            # print('time handle_paf_and_heat:', time2 - time1)
            # --- Step4: both original image and flipped image
            # to_plot, canvas, candidate, connected_limbs, subset = decode_pose(oriImg, param, heatmap, paf, viz=viz, pp=b_postprocessing, solid=b_solid)
            # ---- only flipped image
            # to_plot, canvas, candidate, connected_limbs, subset = decode_pose(oriImg, param, fheat, fpaf, viz=viz)
        else:
            # ---- Step1: Get results of original image
            orig_paf, orig_heat = get_outputs(multiplier, oriImg, model, preprocess)
            heatmap = orig_heat
            paf = orig_paf

        # viz_GT = True
        # viz_Predict = True
        viz_Predict = False
        # ------------ 特征输出可视化
        # if viz_GT:
        #     viz_featuremaps(_heatmap_target.numpy()[0, :, :, :], channelfirst=True, title='HeatMap - GT')
        #     # viz_featuremaps(gt2.cpu().data.numpy()[0, :, :, :], channelfirst=True, title='HeatMap - GT_mask')
        #     viz_featuremaps(gt1.cpu().data.numpy()[0, :, :, :], channelfirst=True, title='PAF - GT')

        # if img_name=='Explorer_HD720_SN14021_15-48-15.jpg':
        #     viz_Predict = True

        if viz_Predict:
            # viz_featuremaps(heatmap.cpu().data.numpy()[0, :, :, :], channelfirst=True, title='HeatMap - predict')
            # viz_featuremaps(paf.cpu().data.numpy()[0, :, :, :], channelfirst=True, title='PAF - predict')
            viz_featuremaps(heatmap[:, :, :], channelfirst=False, title='HeatMap - predict')
            viz_featuremaps(paf[:, :, :], channelfirst=False, title='PAF - predict')

        # ---- Step4: only original image
        # --- Step4: both original image and flipped image
        to_plot, canvas, candidate, connected_limbs, subset = decode_pose(oriImg, param, heatmap, paf,
                                                                          viz=viz, pp=b_postprocessing, solid=b_solid)
        # to_plot, canvas, candidate, connected_limbs, subset = decode_pose(oriImg, param, orig_heat, orig_paf,
        #                                                                   viz=viz, pp=b_postprocessing, solid=b_solid)

        if vis_dir is not None:
            dst_name = img_name[:-4] + '_pp' + '.jpg' if b_postprocessing else img_name
            vis_path = os.path.join(vis_dir, dst_name)
            cv2.imwrite(vis_path, canvas)
        # subset indicated how many peoples foun in this image.

        if save_kp_2_txt:
            # ------ get ground truth
            gt_keypoints_arr = np.array(val_json[json_id]['joint_self'])[:, :-1].astype(np.int32)[np.newaxis, :]
            if len(val_json[json_id]['joint_others']) > 0:
                gt_other = np.array(val_json[json_id]['joint_others'])[:, :, :-1].astype(np.int32)
                gt_keypoints_arr = np.concatenate((gt_keypoints_arr, gt_other), axis=0)
            # results_list.append(gt_keypoints_arr)

            # ------- save all results to txt
            savekp2txt(val_count, results_list, gt_keypoints_arr, img_name, candidate, connected_limbs, subset)

            # ------- save coordinates of the pallets
            save_root='toian'
            if save_root is not None:
                if not os.path.exists(save_root):
                    os.mkdir(save_root)
                    print('mkdir:', save_root)
            # --- copy source image
            copyfile(img_path, os.path.join(save_root, img_name))
            # --- save coordinates of the pallets
            savep2txt(gt_keypoints_arr, img_name, candidate, connected_limbs, subset, save_root)
        #
        # print(candidate, connected_limbs, subset)
        # ----- Second best location to add post-processing
        # post-processing --- 1, 当有3个具有较高置信度的关键点时，来推测第三个点
        # pallet_post(candidate, connected_limbs, subset)

        # append_result(val_json[i]['img_paths'], subset, candidate, outputs)
        append_result(json_id, subset, candidate, outputs)
        # print('append: val_count', val_count, ' img_id:', img_id, 'len(subset)', len(subset))

        # cv2.imshow('test', to_plot)
        # cv2.waitKey(0)
    if save_kp_2_txt:
        with open('kik_keypoints_20190109-results-toian.txt', 'w') as f:  # TODO, change file name
            for item in results_list:
                f.write("%s\n" % item)
    toc = time.time()
    detect_time = toc - tic
    print('Processed {} images within {:.2f} s: {:.2f} FPS'.format(val_count, detect_time, 1.0 * val_count / detect_time))
    print('len(outputs):', len(outputs))

    # print("Total number of validation images {}".format(val_count))
    # Eval and show the final result!
    # return eval_kik_OKS(outputs=outputs, json_path=json_path, imgIds=img_ids)
    return eval_kik_Eud(outputs=outputs, json_data=val_json)
