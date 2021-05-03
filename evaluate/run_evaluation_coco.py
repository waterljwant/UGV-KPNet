import unittest
import torch
from evaluate.coco_eval import run_eval
from network.rtpose_vgg import get_model, use_vgg
from torch import load

#Notice, if you using the 
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    # weight_name = './network/weight/best_pose.pth'
    weight_name = '/home/jie/kiktech-seg/OpenPoseV1/network/weight/pose_model.pth'
    state_dict = torch.load(weight_name)
    model = get_model(trunk='vgg19')

    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()

    model.eval()
    model.float()
    model = model.cuda()
    
    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
    # this repo used 'vgg' preprocess
    run_eval(image_dir='/home/jie/dataset/COCO/images/',
             anno_dir='/home/jie/dataset/COCO/',
             vis_dir='./coco_vis',
             image_list_txt='image_info_val2014_1k.txt',
             model=model, preprocess='vgg')
