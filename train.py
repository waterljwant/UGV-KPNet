import argparse
import time
import datetime
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

from network.rtpose_vgg import get_model, use_vgg
from network import rtpose_shufflenetV2

from training.datasets.ugvkp import get_loader
from network.utility import resample

import UGV_config as CF

parser = argparse.ArgumentParser(description='PyTorch rtpose Training')

parser.add_argument('--json_path', action='append', metavar='PATH', help='path to where json stored')

parser.add_argument('--data_dir', default='', type=str, metavar='DIR', help='path to where coco images stored')
parser.add_argument('--mask_dir', default=' ', type=str, metavar='DIR', help='path to where coco images stored')

parser.add_argument('--logdir', default='./logs', type=str, metavar='DIR', help='path to where tensorboard log restore')

parser.add_argument('--saved_model', default='./network/weight/robot_models.pth', type=str,
                    metavar='PATH', help='path to where kik model to store')

parser.add_argument('--model_path', default='./network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved')

parser.add_argument('--classification', default=False, type=bool, help='classification or regression')

parser.add_argument('--data_augment', default=False, type=bool, help='data augment for training')

parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')

parser.add_argument('--multistage', default=0, type=int, metavar='N', help='number 0f refine stages')
parser.add_argument('--samplenumber', default=72, type=int, metavar='N', help='number 0f refine stages')

parser.add_argument('--gpu_ids', dest='gpu_ids', help='which gpu to use', nargs="+", default=[0], type=int)
parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--batch_size_val', default=4, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--workers_val', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resample', default=False, type=bool, help='data augment for training')

parser.add_argument('--weight-decay', '--wd', default=0.0004, type=float, metavar='W',
                    help='weight decay(default:1e-4)')

parser.add_argument('--lr', '--learning-rate', default=0.2, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

parser.add_argument('--width_mult', default=1.0, type=float, metavar='W', help='width_mult')

parser.add_argument('--nesterov', dest='nesterov', action='store_true')

parser.add_argument('-o', '--optim', default='sgd', type=str)
# Device options

parser.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')

parser.add_argument('--model', default='shufflenet', type=str, help='network backbne')
parser.add_argument('--head', default=None, type=str, help='network head')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)

print('GPU:', args.gpu_ids, ' resample:', args.resample, ' classification', args.classification)


STAGE = args.multistage + 1


def build_names():
    names_paf = []
    names_ht = []
    for j in range(STAGE):
        names_paf.append('loss_stage%d_paf' % (j))
        names_ht.append('loss_stage%d_heatmap' % (j))
    # print(names_paf)
    # print(names_ht)
    return names_paf, names_ht


def _get_loss(saved_for_loss, heat_temp, heat_weight, vec_temp, vec_weight):
    #   get_loss(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    # criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=args.gpu_ids)
    total_loss = 0

    pred1 = saved_for_loss[0] * vec_weight  # [40, 8, 46, 46], [40, 8, 46, 46]
    """
    print("pred1 sizes")
    print(saved_for_loss[2*j].data.size())
    print(vec_weight.data.size())
    print(vec_temp.data.size())
    """
    gt1 = vec_temp * vec_weight  # [40, 8, 46, 46], [40, 8, 46, 46]

    pred2 = saved_for_loss[1] * heat_weight  # [40, 5, 46, 46], [40, 4, 46, 46]
    gt2 = heat_weight * heat_temp  # [40, 4, 46, 46], [40, 5, 46, 46]
    """
    print("pred2 sizes")
    print(saved_for_loss[2*j+1].data.size())
    print(heat_weight.data.size())   # [40, 4, 46, 46]
    print(heat_temp.data.size())     # [40, 5, 46, 46]
    """

    # resample_mask = np.zeros(gt2.shape, dtype=np.float)
    # bs, c, h, w = gt2.shape
    #
    # y_true_0123_dix = np.where(gt2 == 0)
    # y_true_4_dix = np.where(gt2 == 1)

    # Compute losses
    loss1 = criterion(pred1, gt1)
    loss2 = criterion(pred2, gt2)

    total_loss += loss1 * 0.5  # *0.1  # PAF # TODO
    total_loss += loss2  # Heatmap
    # print(total_loss)
    # print('PAF', loss1.item(), 'HeatMap', loss2.item())

    # Get value from Variable and save for log
    saved_for_log['paf'] = loss1.item()
    saved_for_log['heatmap'] = loss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def get_loss(saved_for_loss, heat_temp, heat_weight, vec_temp, vec_weight):
    #   get_loss(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)
    saved_for_log = OrderedDict()

    criterion = nn.MSELoss(size_average=True).cuda()

    criterion_cls = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=args.gpu_ids)
    total_loss = 0

    names_paf, names_ht = build_names()

    # print(vec_weight.data.size(), saved_for_loss[0].data.size())
    # print('len(saved_for_loss):', len(saved_for_loss))
    for j in range(len(saved_for_loss) // 2):
        # print('j:', j)
        # for j in range(len(saved_for_loss)):
        # PAF
        # print('vec_weight.shape:', vec_weight.shape)
        # print("pred1 sizes")
        # print(saved_for_loss[2 * j].data.size())
        # print("pred2 sizes")
        # print(saved_for_loss[2 * j + 1].data.size())

        pred1 = saved_for_loss[2 * j] * vec_weight  # [40, 8, 46, 46], [40, 8, 46, 46]
        gt1 = vec_temp * vec_weight  # [40, 8, 46, 46], [40, 8, 46, 46]
        loss1 = criterion(pred1, gt1)
        """
        print("pred1 sizes")
        print(saved_for_loss[2*j].data.size())
        print(vec_weight.data.size())
        print(vec_temp.data.size())
        """

        """
        print("pred2 sizes")
        print(saved_for_loss[2*j+1].data.size())
        print(heat_weight.data.size())   # [40, 4, 46, 46]
        print(heat_temp.data.size())     # [40, 5, 46, 46]
        """

        # Compute losses
        if args.classification:
            pred2 = saved_for_loss[2 * j + 1]
            gt2 = heat_temp
            loss2 = criterion_cls(pred2, gt2)
        else:
            pred2 = saved_for_loss[2 * j + 1] * heat_weight
            gt2 = heat_temp * heat_weight
            loss2 = criterion(pred2, gt2)

        total_loss += loss1  # * 0.1  # TODO weight
        total_loss += loss2  # * 10  # TODO weight
        # print(total_loss)
        # print('PAF', loss1.item(), 'HeatMap', loss2.item())

        # Get value from Variable and save for log
        # names_paf = 'loss_stage%d_paf' % (j)
        # names_ht = 'loss_stage%d_heatmap' % (j)

        # saved_for_log['paf'] = loss1.item()
        # saved_for_log['heatmap'] = loss2.item()
        # print('names_paf[j]', names_paf[j])
        # print('names_ht[j]', names_ht[j])
        saved_for_log[names_paf[j]] = loss1.item()
        saved_for_log[names_ht[j]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}

    # meter_dict['paf'] = AverageMeter()
    # meter_dict['heatmap'] = AverageMeter()

    names_paf, names_ht = build_names()
    for paf in names_paf:
        meter_dict[paf] = AverageMeter()
    for ht in names_ht:
        meter_dict[ht] = AverageMeter()

    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()

    # switch to train mode
    model.train()
    neg_rate = min(max(1. - (epoch + 1) / args.samplenumber, 0.), 0.95)
    if args.resample:
        print(
            'Epoch: {}, ignore_rate:{:.4f}, Learning rate:{}'.format(epoch, neg_rate, optimizer.param_groups[0]['lr']))
    else:
        print('Epoch: {}, Learning rate:{}'.format(epoch, optimizer.param_groups[0]['lr']))

    end = time.time()
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask, img_paths) in enumerate(train_loader):
        # measure data loading time
        # writer.add_text('Text', 'text logged at step:' + str(i), i)

        # for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)
        data_time.update(time.time() - end)

        # --------------------------------- resample ------------------------------------
        if args.resample and neg_rate > 0:
            heat_mask, paf_mask = resample(heatmap_target, heat_mask, paf_target, paf_mask, neg_rate)

        # print('img.shape:', img.shape, 'heatmap_target:', heatmap_target.shape)
        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        heat_mask = heat_mask.cuda()
        paf_target = paf_target.cuda()
        paf_mask = paf_mask.cuda()

        # compute output
        # _,saved_for_loss = model(img)
        saved_for_loss = model(img)  # [PAF, HEAT]

        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)

        for name, _ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # TODO weight clip
        # c = 0.01
        # for p in model.parameters():
        #     p.data.clamp_(-c, c)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string += 'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string += 'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.val:.5f} ({loss.avg:.5f})\t'.format(name=name, loss=value)
            print(print_string)

    return losses.avg


def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}
    # meter_dict['paf'] = AverageMeter()
    # meter_dict['heatmap'] = AverageMeter()

    names_paf, names_ht = build_names()
    for paf in names_paf:
        meter_dict[paf] = AverageMeter()
    for ht in names_ht:
        meter_dict[ht] = AverageMeter()

    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    # model.train()
    model.eval()

    end = time.time()
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask, img_paths) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        heat_mask = heat_mask.cuda()
        paf_target = paf_target.cuda()
        paf_mask = paf_mask.cuda()

        # compute output
        # _,saved_for_loss = model(img)
        saved_for_loss = model(img)

        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
                                             paf_target, paf_mask)

        for name, _ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss.item(), img.size(0))
        # losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_string = 'Val Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string += 'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


print("Loading dataset...")
# load data

train_data = get_loader(args.json_path, args.data_dir, args.mask_dir, inp_size=None, feat_stride=8,
                        preprocess='rtpose', batch_size=args.batch_size, params_transform=CF.params_transform,
                        shuffle=True, training=True, num_workers=args.workers, aug=args.data_augment,
                        classification=args.classification)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, inp_size=None, feat_stride=8,
                        preprocess='rtpose', batch_size=args.batch_size_val, params_transform=CF.params_transform,
                        shuffle=False, training=False, num_workers=args.workers,
                        classification=args.classification)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
model_name = args.model
width_mult = args.width_mult
print('width_mult:{}'.format(width_mult))
if model_name == 'vgg19':  # --- VGG19
    model = get_model(trunk='vgg19', numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS)
    preprocess = 'vgg'
elif model_name == 'shufflenet':  # --- ShuffleNet
    model = rtpose_shufflenetV2.Network(width_multiplier=1.0, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                        multistage=args.multistage)
    preprocess = 'rtpose'
elif model_name == 'LWshufflenet':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_baseline':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_baseline',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_baseline_v1':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_baseline_v1',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_baseline_v1_cat':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_baseline_v1_cat',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_v1_16_cat':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_v1_16_cat',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_HR_cat':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_HR_cat',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_HR_catv2':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_HR_catv2',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_HRv2':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_HRv2',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_baseline_v2':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_baseline_v2',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_baseline_v3':  # ShuffleNetV2 baseline
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_baseline_v3',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_SingleASPP':  # --- ShuffleNetV2 + ASPP
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_SingleASPP',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_SingleASPP192':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_SingleASPP192',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_MultiASPP192':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_MultiASPP192',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_MultiASPP':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_MultiASPP',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_mscat':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_mscat',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'LWShuffleNetV2_msadd':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='LWShuffleNetV2_msadd',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'ShuffleNetV2_cat':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='ShuffleNetV2_cat',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'ShuffleNetV2_Adaptive_cat':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='ShuffleNetV2_Adaptive_cat',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'ShuffleNetV2_Adaptive_catV2':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='ShuffleNetV2_Adaptive_catV2',
                                          head=args.head)
    preprocess = 'rtpose'
elif model_name == 'ShuffleNetV2_add':  # --- Light weight ShuffleNet
    model = rtpose_shufflenetV2.LWNetwork(width_mult=width_mult, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
                                          multistage=args.multistage, backbone='ShuffleNetV2_add',
                                          head=args.head)
    preprocess = 'rtpose'
else:
    print('Please check the model name.')
    exit(0)
print('Network backbone:{}'.format(model_name))

# model = rtpose_shufflenetV2.Network(width_multiplier=1.0, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
#                                     multistage=args.multistage)  # Mod by Jie.

# model = rtpose_shufflenetV2.LWNetwork(width_mult=1.0, numkeypoints=CF.NUM_KEYPOINTS, numlims=CF.NUM_LIMBS,
#                                       multistage=args.multistage)  # Mod by Jie.
# model = encoding.nn.DataParallelModel(model, device_ids=args.gpu_ids)

model = torch.nn.DataParallel(model).cuda()

# --- resume
if args.resume is not None:
    # --- resume
    model.load_state_dict(torch.load(args.resume), strict=False)
    print('Resumed from model:', args.resume)

writer = SummaryWriter(log_dir=args.logdir)

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                 threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

best_val_loss = np.inf

print('Start to train')
train_time_start = datetime.datetime.now()
for epoch in range(args.epochs):

    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch)

    # for param_group in optimizer.param_groups:
    #     print('lr: ', param_group['lr'])

    # evaluate on validation set
    val_loss = validate(valid_data, model, epoch)
    print('Epoch:', epoch, ' train_loss', train_loss.data.cpu(), ' val_loss:', val_loss)

    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
                                             'val loss': val_loss}, epoch)
    lr_scheduler.step(val_loss)
    # --- learning rate
    # lr_scheduler.step()
    # for param_group in optimizer.param_groups:
    #     print('learning rate:{}'.format(param_group['lr']))
    # decs_str = 'Training epoch {}/{}'.format(epoch + 1, args.epochs)

    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)

    if is_best:
        print('Get the best at {} epoch'.format(epoch))
        torch.save(model.state_dict(), args.saved_model[:-4] + '_best.pth')

# ----- ----- ----- -----training finished
print('Training finished in: {}'.format(datetime.datetime.now() - train_time_start))

# save the last model
torch.save(model.state_dict(), args.saved_model)

writer.export_scalars_to_json(os.path.join(args.model_path, "tensorboard/all_scalars.json"))
writer.close()


