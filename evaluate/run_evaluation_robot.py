#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import datetime
from evaluate.ugv_eval import run_eval, show_GT
from network.rtpose_vgg import get_model
from network import rtpose_shufflenetV2

import UGV_config as CF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

json_path = list()
# 1280 x 720
# json_path.append('/home/jie/kikprogram/robot/dataset/robot_keypoints125.json')
# 640 x 360
# json_path.append('/home/jie/kikprogram/robot/dataset/robot_keypoints125_1-4.json')
# json_path.append('/home/jie/kikprogram/robot/dataset/robot_keypoints_20190410_1-4.json')
# json_path.append('/home/jie/kikprogram/robot/dataset/robot_keypoints_20190421_640x360.json')
json_path.append('/home/jie/kikprogram/robot/dataset/robot_keypoints_20190426_640x360.json')

strFlag = []
# strFlag.append('flip')  # TODO no flip in robot kp.
# strFlag.append('multi-scale')
strFlag.append('post-processing')
# strFlag.append('solid')

# ------------------------------------------------------------------------------------------------------------
# ShuffleNet
model_name='shufflenet'
# baseline  2019-04-27
# multistage=0
# weight_name='../network/weight/shufflenet_robot_ms0_baseline_v1.0.pth'  # P81.36% R63.58%

multistage=0
# weight_name='../network/weight/shufflenet_robot_ms0-v0.1.0.pth'       # Precision:	85.92% Recall:	80.79%
# weight_name='../network/weight/shufflenet_robot_ms0-v0.1.0_best.pth'  # Precision:	82.31% Recall:	80.13%
# weight_name='../network/weight/shufflenet_robot_ms0-v0.1.1.pth'       # Precision:	90.60% Recall:	89.40%
# weight_name='../network/weight/shufflenet_robot_ms0-v0.1.1_best.pth'  # Precision:	91.95% Recall:	90.73%
# weight_name='../network/weight/shufflenet_robot_ms0-v0.1.2.pth'         # Precision:	90.85% Recall:	92.05%
# weight_name='../network/weight/shufflenet_robot_ms0-v0.1.2_best.pth'  # Precision:	90.79% Recall:	91.39%
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_ms0-try.pth'  #
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_ms0-try_best.pth'  # nan

#
# multistage=1
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot-v0.1_ms1.pth'
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_ms1-v0.2.pth'
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_ms1-v0.1.0-tmp_best.pth'  # nan
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_m10-v0.1.baseline.pth'  # nan
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_ms1-v0.1.baseline.pth'  # nan
#
# multistage=2
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot-v0.1_ms2_best.pth'
# weight_name='/home/jie/kikprogram/robot/OpenPoseV1/network/weight/shufflenet_robot_ms2-v0.2_best.pth'

# vis_dir = './robot_vis-1280x720_test'
vis_dir = '../../../Desktop/IET/robot/OpenPoseV1/evaluate/robot_vis-640x360_test-ms'
# vis_dir = './robot_vis-640x360_ms0-v0.1.2'

# ------------------------------------------------------------------------------------------------------------
# VGG19
# model_name='vgg19'

# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125-100-lr0.1.pth'  # --NAN
# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125-lr0.1_2.pth'


#
# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125_1-4_e50.pth'

# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125-500.pth'

# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125-100.pth'  #
# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125-100-lr0.1.pth'

# weight_name = '/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_robot_125-100-lr1.0-resample.pth'



# json_path.append('/home/jie/dataset/kik-keypoints/kik_keypoints_20190125-front-view.json')
# weight_name='/home/jie/kiktech-seg/robot/OpenPoseV1/network/weight/vgg19_kikrobot_125-20.pth'
# vis_dir = None
# vis_dir = './robot_vis-dataset125_1-4_e40'


# ------------------------------------------------------------------------------
tic = datetime.datetime.now()

# ------------------------------------------------------------------------------
# 将 keypoints 以编号0123的形式，在图片上标记出来
# vis_GT = 'kik_vis_0125_GT_VAL'
# show_GT(json_path, vis_GT, 'VAL')  # VAL or TRAIN or None for both


if vis_dir is not None:
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
        print('mkdir:', vis_dir)
print('save vis images to:', vis_dir)


# valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, inp_size=None, feat_stride=8,
#                         preprocess='rtpose', batch_size=args.batch_size,  params_transform=CF.params_transform,
#                         shuffle=False, training=False, num_workers=args.workers,
#                         classification=args.classification)
# print('val dataset len: {}'.format(len(valid_data.dataset)))

# ------------------------------------------------------------------------------
with torch.autograd.no_grad():

    if model_name == 'vgg19':            # --- VGG19
        model = get_model(trunk='vgg19', numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS)
        preprocess = 'vgg'
    elif model_name == 'shufflenet':     # --- ShuffleNet
        model = rtpose_shufflenetV2.Network(width_multiplier=1.0, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS, multistage=multistage)
        preprocess = 'rtpose'
    else:
        print('Please check the model name.')
        exit(0)
    print('Network backbone:{}'.format(model_name))

    # this path is with respect to the root of the project
    state_dict = torch.load(weight_name)

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)

    model.eval()
    model.float()
    model = model.cuda()

    run_eval(json_path=json_path,
             model=model,
             preprocess=preprocess,
             vis_dir=vis_dir,
             flag=strFlag)

toc = datetime.datetime.now()
print('Spend time:', toc-tic)