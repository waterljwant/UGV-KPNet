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

# multistage=0
# weight_name='../network/weight/shufflenet_robot_ms0.pth'

#
multistage=2
weight_name='../network/weight/shufflenet_robot_ms2.pth'


vis_dir = './result_vis'


# ------------------------------------------------------------------------------
tic = datetime.datetime.now()


if vis_dir is not None:
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
        print('mkdir:', vis_dir)
print('save vis images to:', vis_dir)

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
