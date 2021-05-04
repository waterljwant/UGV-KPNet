#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import caffe
    from caffe import layers as L
    from caffe import params as P
except ImportError:
    pass

from network import slim
from network.slim import g_name


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [pafs, heatmaps]


class BasicBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, stride, downsample, dilation):
        super(BasicBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        self.downsample = downsample
        channels = out_channels//2
        if not self.downsample and self.stride==1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(
                slim.conv_bn_relu(name + '/conv1', channels, channels, 1),
                slim.conv_bn(name + '/conv2',
                    channels, channels, 3, stride=stride,
                    dilation=dilation, padding=dilation, groups=channels),
                slim.conv_bn_relu(name + '/conv3', channels, channels, 1),
            )
        else:
            self.conv = nn.Sequential(
                slim.conv_bn_relu(name + '/conv1', in_channels, channels, 1),
                slim.conv_bn(name + '/conv2',
                    channels, channels, 3, stride=stride,
                    dilation=dilation, padding=dilation, groups=channels),
                slim.conv_bn_relu(name + '/conv3', channels, channels, 1),
            )
            self.conv0 = nn.Sequential(
                slim.conv_bn(name + '/conv4',
                    in_channels, in_channels, 3, stride=stride,
                    dilation=dilation, padding=dilation, groups=in_channels),
                slim.conv_bn_relu(name + '/conv5', in_channels, channels, 1),
            )
        self.shuffle = slim.channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if not self.downsample:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)

    def generate_caffe_prototxt(self, caffe_net, layer):
        if self.stride == 1:
            layer_x1, layer_x2 = L.Slice(layer, ntop=2, axis=1, slice_point=[self.in_channels//2])
            caffe_net[self.g_name + '/slice1'] = layer_x1
            caffe_net[self.g_name + '/slice2'] = layer_x2
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer_x2)
        else:
            layer_x1 = slim.generate_caffe_prototxt(self.conv0, caffe_net, layer)
            layer_x2 = slim.generate_caffe_prototxt(self.conv, caffe_net, layer)
        layer = L.Concat(layer_x1, layer_x2, axis=1)
        caffe_net[self.g_name + '/concat'] = layer
        layer = slim.generate_caffe_prototxt(self.shuffle, caffe_net, layer)
        return layer


# width_multiplier=1.0
# Params: 1266935, 1.27M
# Flops:  6186360000.0, 6.19G
# width_multiplier=0.5
# Params: 355123, 355.12K
# Flops:  1783550400.0, 1.78G
class Network(nn.Module):
    def __init__(self, width_multiplier, numkeypoints=18, numlims=19, multistage=0):  # Mod by Jie
        super(Network, self).__init__()
        width_config = {
            0.25: (24, 48, 96, 512),
            0.33: (32, 64, 128, 512),
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (244, 488, 976, 2048),
        }
        #     self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        # elif width_mult == 1.0:
        #     self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        # elif width_mult == 1.5:
        #     self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        # elif width_mult == 2.0:
        #     self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]

        width_config = width_config[width_multiplier]
        in_channels = 24

        self.multistage = multistage

        # outputs, stride, dilation, blocks, type
        self.network_config = [
            g_name('data/bn', nn.BatchNorm2d(3)),
            slim.conv_bn_relu('stage1/conv', 3, in_channels, 3, 2, 1),
            g_name('stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)),
            (width_config[0], 2, 1, 4, 'b'),
            (width_config[1], 1, 1, 8, 'b'),  # x16
            (width_config[2], 1, 1, 4, 'b'),  # x32
            slim.conv_bn_relu('conv5', width_config[2], width_config[3], 1)
        ]

        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            if stride==2:
                downsample=True
            stage_prefix = 'stage_{}'.format(i - 1)
            blocks = [BasicBlock(stage_prefix + '_1', in_channels,
                out_channels, stride, downsample, dilation)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(stage_prefix + '_{}'.format(i + 1),
                    out_channels, out_channels, 1, False, dilation))
            self.network += [nn.Sequential(*blocks)]

            in_channels = out_channels
        self.network = nn.Sequential(*self.network)

        self.paf = nn.Conv2d(width_config[3], numlims*2, 1)           # Mod by Jie
        self.heatmap = nn.Conv2d(width_config[3], numkeypoints+1, 1)  # Mod by Jie. channels: background + numkeypoints
        # self.th_paf = nn.Tanh()
        # self.sf_heatmap = nn.Softmax(dim=1)

        if self.multistage > 0:
            num_channels = 128
            self.refinement_stages = nn.ModuleList()
            ch = numlims * 2 + numkeypoints + 1
            for i in range(self.multistage):
                # self.refinement_stages.append(nn.Conv2d(in_channels=ic1 + ch, out_channels=ch, kernel_size=3, padding=1))
                self.refinement_stages.append(RefinementStage(in_channels=width_config[3] + ch,
                                                              out_channels=num_channels,
                                                              num_heatmaps=numkeypoints + 1,
                                                              num_pafs=numlims * 2))

        for name, m in self.named_modules():
            if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def trainable_parameters(self):
        parameters = [
            {'params': self.cls_head_list.parameters(), 'lr_mult': 1.0},
            {'params': self.loc_head_list.parameters(), 'lr_mult': 1.0}
        ]
        for i in range(len(self.network)):
            lr_mult = 0.1 if i in (0, 1, 2, 3, 4, 5) else 1
            parameters.append(
                {'params': self.network[i].parameters(), 'lr_mult': lr_mult}
            )
        return parameters

    def forward(self, x):
        x = self.network(x)
        # print('x.size()', x.size())  # [1, 1024, 60, 80]
        PAF = self.paf(x)
        HEAT = self.heatmap(x)
        # PAF = self.th_paf(PAF)
        # HEAT = self.sf_heatmap(HEAT)   # TODO

        stages_output = [PAF, HEAT]
        if self.multistage > 0:
            # stages_output=[]
            for refinement_stage in self.refinement_stages:
                # stages_output.append( refinement_stage(torch.cat([x, stages_output[-2], stages_output[-1]], dim=1)) )
                stages_output.extend( refinement_stage(torch.cat([x, stages_output[-2], stages_output[-1]], dim=1)) )

        return stages_output


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, c_in, c_out, dilation=[1, 2, 4, 8, 16], global_pool=False):
        super(ASPP, self).__init__()
        print('ASPP: c_in:{}, c_out:{}, dilation:{}'.format(c_in, c_out, dilation))

        self.global_pool = global_pool
        self.n = len(dilation)

        d = dilation
        k = [1, 3, 3, 3, 3]
        p = k[0] // 2 * d[0]

        self.aspp0 = _ASPPModule(c_in, c_out, k[0], padding=p, dilation=d[0])
        p = k[1] // 2 * d[1]
        self.aspp1 = _ASPPModule(c_in, c_out, k[1], padding=p, dilation=d[1])
        p = k[2] // 2 * d[2]
        self.aspp2 = _ASPPModule(c_in, c_out, k[2], padding=p, dilation=d[2])

        if self.n > 3:
            p = k[3] // 2 * d[3]
            self.aspp3 = _ASPPModule(c_in, c_out, k[3], padding=p, dilation=d[3])
        if self.n > 4:
            p = k[4] // 2 * d[4]
            self.aspp4 = _ASPPModule(c_in, c_out, k[4], padding=p, dilation=d[4])

        if self.global_pool:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_in, c_out, 1, stride=1, bias=False))

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        y = torch.cat((x0, x1, x2), dim=1)
        if self.n > 3:
            x3 = self.aspp3(x)
            y = torch.cat((y, x3), dim=1)
        if self.n > 4:
            x4 = self.aspp4(x)
            y = torch.cat((y, x4), dim=1)
        if self.global_pool:
            x_ = self.global_avg_pool(x)
            x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
            y = torch.cat((y, x_), dim=1)
        return y


class ResidualAdapter(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=1, dilation=1, residual=False):
        super(ResidualAdapter, self).__init__()
        print('ResidualAdapter: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))
        self.residual = residual
        self.conv_in = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        p = kernel // 2 * dilation
        self.conv = nn.Conv2d(c, c, kernel_size=kernel, stride=1, padding=p, dilation=dilation, bias=False)
        self.adapter = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv_out = nn.Sequential(
            nn.Conv2d(c, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(c_out))

        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x0 = self.conv_in(x)
        y = self.conv(x0) + self.adapter(x0)  # Element-wise add
        y = self.conv_out(y)
        y = self.relu(y + x) if self.residual else self.relu(y)
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# Remove relu
class ResidualAdapterV2(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=1, dilation=1, residual=False):
        super(ResidualAdapterV2, self).__init__()
        print('ResidualAdapterV2: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))
        self.residual = residual
        self.conv_in = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(c),
            # nn.ReLU(inplace=True)
        )
        p = kernel // 2 * dilation
        self.conv = nn.Conv2d(c, c, kernel_size=kernel, stride=1, padding=p, dilation=dilation, bias=False)
        self.adapter = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv_out = nn.Sequential(
            nn.Conv2d(c, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        # self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x0 = self.conv_in(x)
        y = self.conv(x0) + self.adapter(x0)  # Element-wise add
        y = self.conv_out(y)
        # y = self.relu(y + x) if self.residual else self.relu(y)
        y = y + x if self.residual else y
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResidualAdapterV3(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=1, dilation=1, groups=1, residual=False):
        super(ResidualAdapterV3, self).__init__()
        print('ResidualAdapterV3: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))
        self.residual = residual
        self.conv_in = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, dilation=1, groups=groups, bias=False),
            nn.BatchNorm2d(c),
            # nn.ReLU(inplace=True)
        )
        p = kernel // 2 * dilation
        self.conv = nn.Conv2d(c, c, kernel_size=kernel, stride=1, padding=p, dilation=dilation, groups=c, bias=False)
        self.adapter = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, dilation=1, groups=c, bias=True)
        self.conv_out = nn.Sequential(
            nn.Conv2d(c, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        # self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x0 = self.conv_in(x)
        y = self.conv(x0) + self.adapter(x0)  # Element-wise add
        y = self.conv_out(y)
        # y = self.relu(y + x) if self.residual else self.relu(y)
        y = y + x if self.residual else y
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResidualAdapterMix(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=1, dilation=1, residual=False):
        super(ResidualAdapterMix, self).__init__()
        print('ResidualAdapterMix: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))
        self.residual = residual
        self.conv_in = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            # nn.BatchNorm2d(c),
            # nn.ReLU(inplace=True)
        )
        p = kernel // 2 * dilation
        self.conv = nn.Conv2d(c, c, kernel_size=kernel, stride=1, padding=p, dilation=dilation, bias=False)

        self.basis_channel = c
        # self.adapter = nn.Conv2d(c_in, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.adapter = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            # nn.Tanh(),  # [-1, 1]
            nn.Sigmoid(),  # [0, 1]
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(c, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            # nn.BatchNorm2d(c_out)
        )

        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x0 = self.conv_in(x)
        y = self.conv(x0)

        # mx = self.adapter(x)  # (BS, 1, basis_channel, H, W)
        mx = self.adapter(x0)  # (BS, 1, basis_channel, H, W)
        mx = mx.expand(-1, self.basis_channel, -1, -1)  # (BS, basis_channel, H, W)
        y = torch.mul(mx, y)

        y = self.conv_out(y)
        y = self.relu(y + x) if self.residual else self.relu(y)
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LWASPP(nn.Module):
    def __init__(self, c_in, c, c_out, dilation=[1, 2, 4, 8, 16], global_pool=False):
        super(LWASPP, self).__init__()
        print('LWASPP: c_in:{}, c:{}, c_out:{}, dilation:{}'.format(c_in, c, c_out, dilation))

        self.global_pool = global_pool
        self.n = len(dilation)

        d = dilation
        k = [1, 3, 3, 3, 3]
        p = k[0] // 2 * d[0]

        self.conv_in = nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, bias=False)

        self.aspp0 = _ASPPModule(c, c_out, k[0], padding=p, dilation=d[0])
        p = k[1] // 2 * d[1]
        self.aspp1 = _ASPPModule(c, c_out, k[1], padding=p, dilation=d[1])
        p = k[2] // 2 * d[2]
        self.aspp2 = _ASPPModule(c, c_out, k[2], padding=p, dilation=d[2])

        if self.n > 3:
            p = k[3] // 2 * d[3]
            self.aspp3 = _ASPPModule(c, c_out, k[3], padding=p, dilation=d[3])
        if self.n > 4:
            p = k[4] // 2 * d[4]
            self.aspp4 = _ASPPModule(c, c_out, k[4], padding=p, dilation=d[4])

        if self.global_pool:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c_out, 1, stride=1, bias=False))

    def forward(self, x):
        _x = self.conv_in(x)
        x0 = self.aspp0(_x)
        x1 = self.aspp1(_x)
        x2 = self.aspp2(_x)
        y = torch.cat((x0, x1, x2), dim=1)
        if self.n > 3:
            x3 = self.aspp3(_x)
            y = torch.cat((y, x3), dim=1)
        if self.n > 4:
            x4 = self.aspp4(_x)
            y = torch.cat((y, x4), dim=1)
        if self.global_pool:
            x_ = self.global_avg_pool(_x)
            x_ = F.interpolate(x_, size=_x.size()[2:], mode='trilinear', align_corners=True)
            y = torch.cat((y, x_), dim=1)
        return y


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class InvertedResidualV2(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidualV2, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:  # output channel = inp + oup // 2
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                # nn.ReLU(inplace=True),  # TODO: remove ReLU
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:  # output channel = oup // 2  * 2
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                # nn.ReLU(inplace=True), # TODO: remove ReLU
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2_v1(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_v1, self).__init__()
        print('ShuffleNetV2_v1, using InvertedResidualV2.')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidualV2(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidualV2(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)  # 1/4
        x = self.stage_2(x)   # 1/8
        x = self.stage_3(x)   # 1/16
        x = self.stage_4(x)   # 1/32
        return x


class LWShuffleNetV2_v1_cat(nn.Module):
    def __init__(self, width_mult=0.5):
        super(LWShuffleNetV2_v1_cat, self).__init__()
        print('LWShuffleNetV2_v1_cat, using InvertedResidualV2. 1/16, cat')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # stride = 2 if idxstage == 0 else 1
                    stride = 2 if idxstage == 0 or idxstage == 2 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidualV2(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidualV2(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        self.upsample4 = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        f1 = self.maxpool(x)  # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/8
        f4 = self.stage_4(f3)   # 1/16

        x_f2 = f2
        x_f3 = f3
        x_f4 = self.upsample4(f4)  # 464/4 = 116

        h_min = min(x_f2.size(2), x_f3.size(2), x_f4.size(2))
        ds2 = x_f2.size(2) - h_min
        ds3 = x_f3.size(2) - h_min
        ds4 = x_f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = x_f2[:, :, ds2//2:x_f2.size(2)-(ds2-ds2//2), :]
        x_f3 = x_f3[:, :, ds3//2:x_f3.size(2)-(ds3-ds3//2), :]
        x_f4 = x_f4[:, :, ds4//2:x_f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        y = torch.cat((x_f2, x_f3, x_f4), dim=1)

        # y = torch.cat((f2, f3, f4), dim=1)  # 116, 232, 464/4
        return y


def _cat2(x1, x2):
    h_min = min(x1.size(2), x2.size(2))
    ds1 = x1.size(2) - h_min
    ds2 = x2.size(2) - h_min
    f1 = x1[:, :, ds1 // 2:x1.size(2) - (ds1 - ds1 // 2), :]
    f2 = x2[:, :, ds2 // 2:x2.size(2) - (ds2 - ds2 // 2), :]
    return torch.cat((f1, f2), dim=1)


def _cat3(x1, x2, x3):
    h_min = min(x1.size(2), x2.size(2), x3.size(2))
    ds1 = x1.size(2) - h_min
    ds2 = x2.size(2) - h_min
    ds3 = x3.size(2) - h_min
    f1 = x1[:, :, ds1 // 2:x1.size(2) - (ds1 - ds1 // 2), :]
    f2 = x2[:, :, ds2 // 2:x2.size(2) - (ds2 - ds2 // 2), :]
    f3 = x3[:, :, ds3 // 2:x3.size(2) - (ds3 - ds3 // 2), :]
    return torch.cat((f1, f2, f3), dim=1)


class LWShuffleNetV2_HR_cat(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_HR_cat, self).__init__()
        print('LWShuffleNetV2_HR_cat, using InvertedResidualV2. 1/16, cat')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            # self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
            self.stage_out_channels = [-1, 24, 64, 128, 256, 512, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        df = 1  # down-sample channel expansion factor
        input_channel_2_1 = self.stage_out_channels[1]
        output_channel_2_1 = self.stage_out_channels[2]
        self.stage_2_1 = nn.Sequential(InvertedResidualV2(input_channel_2_1, output_channel_2_1, 2, 2),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     )  # 1/8
        self.down2_1to3_2 = nn.Sequential(nn.Conv2d(output_channel_2_1, output_channel_2_1 * df, 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channel_2_1 * df),
                                          nn.ReLU(inplace=True)
                                          )

        input_channel_3_1 = output_channel_2_1
        output_channel_3_1 = self.stage_out_channels[3]
        # print('3_1:{}, {}'.format(input_channel_3_1, output_channel_3_1))
        self.stage_3_1 = nn.Sequential(InvertedResidualV2(input_channel_3_1, output_channel_3_1, 1, 2),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     )
        self.down3_1to4_2 = nn.Sequential(nn.Conv2d(output_channel_3_1, output_channel_3_1 * df, 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channel_3_1 * df),
                                          nn.ReLU(inplace=True)
                                          )

        input_channel_3_2 = output_channel_2_1 * df
        output_channel_3_2 = input_channel_3_2
        # print('3_2:{}, {}'.format(input_channel_3_2, output_channel_3_2))
        self.stage_3_2 = nn.Sequential(InvertedResidualV2(input_channel_3_2, output_channel_3_2, 1, 2),
                                       InvertedResidualV2(output_channel_3_2, output_channel_3_2, 1, 1),
                                       InvertedResidualV2(output_channel_3_2, output_channel_3_2, 1, 1),
                                       InvertedResidualV2(output_channel_3_2, output_channel_3_2, 1, 1),
                                       )
        self.up3_2to4_1 = nn.PixelShuffle(2)

        input_channel_4_1 = self.stage_out_channels[3] + output_channel_3_2//4
        output_channel_4_1 = self.stage_out_channels[4]
        # print('4_1:{}, {}'.format(input_channel_4_1, output_channel_4_1))
        self.stage_4_1 = nn.Sequential(InvertedResidualV2(input_channel_4_1, output_channel_4_1, 1, 2),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     )
        self.down4_1to5_2 = nn.Sequential(nn.Conv2d(output_channel_4_1, output_channel_4_1 * df, 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channel_4_1 * df),
                                          nn.ReLU(inplace=True)
                                          )
        # self.down4_1to5_3 = nn.Sequential(nn.Conv2d(output_channel_4_1, output_channel_4_1 * df, 3, 2, 1, bias=False),
        #                                   nn.Conv2d(output_channel_4_1 * df, output_channel_4_1 * (df + df), 3, 2, 1, bias=False),
        #                                   nn.BatchNorm2d(output_channel_4_1 * (df + df)),
        #                                   nn.ReLU(inplace=True)
        #                                   )

        self.down4_1to5_3 = nn.Sequential(
            # nn.Conv2d(output_channel_4_1, output_channel_4_1 * df, 3, 2, 1, bias=False),
            nn.Conv2d(output_channel_4_1 * df, output_channel_4_1 * (df + df), 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel_4_1 * (df + df)),
            nn.ReLU(inplace=True)
        )

        input_channel_4_2 = output_channel_3_2 + output_channel_3_1 * df
        output_channel_4_2 = input_channel_4_2
        # print('4_2:{}, {}'.format(input_channel_4_2, output_channel_4_2))
        self.stage_4_2 = nn.Sequential(InvertedResidualV2(input_channel_4_2, output_channel_4_2, 1, 2),
                                       InvertedResidualV2(output_channel_4_2, output_channel_4_2, 1, 1),
                                       InvertedResidualV2(output_channel_4_2, output_channel_4_2, 1, 1),
                                       InvertedResidualV2(output_channel_4_2, output_channel_4_2, 1, 1),
                                       )
        self.up4_2to5_1 = nn.PixelShuffle(2)
        self.down4_2to5_3 = nn.Sequential(
            nn.Conv2d(output_channel_4_2, output_channel_4_2 * df, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel_4_2 * df),
            nn.ReLU(inplace=True)
            )

        input_channel_5_1 = output_channel_4_1 + output_channel_4_2 // 4
        output_channel_5_1 = self.stage_out_channels[4]
        # print('5_1:{}, {}'.format(input_channel_5_1, output_channel_5_1))
        self.stage_5_1 = nn.Sequential(InvertedResidualV2(input_channel_5_1, output_channel_5_1, 1, 2),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     )

        input_channel_5_2 = output_channel_4_2 + output_channel_4_1 * df
        output_channel_5_2 = input_channel_5_2
        # print('5_2:{}, {}'.format(input_channel_5_2, output_channel_5_2))
        self.stage_5_2 = nn.Sequential(InvertedResidualV2(input_channel_5_2, output_channel_5_2, 1, 2),
                                       InvertedResidualV2(output_channel_5_2, output_channel_5_2, 1, 1),
                                       InvertedResidualV2(output_channel_5_2, output_channel_5_2, 1, 1),
                                       InvertedResidualV2(output_channel_5_2, output_channel_5_2, 1, 1),
                                       )

        input_channel_5_3 = output_channel_4_2 * df + output_channel_4_1 * (df + df)
        output_channel_5_3 = input_channel_5_3
        # print('5_3:{}, {}'.format(input_channel_5_3, output_channel_5_3))
        self.stage_5_3 = nn.Sequential(InvertedResidualV2(input_channel_5_3, output_channel_5_3, 1, 2),
                                       InvertedResidualV2(output_channel_5_3, output_channel_5_3, 1, 1),
                                       InvertedResidualV2(output_channel_5_3, output_channel_5_3, 1, 1),
                                       InvertedResidualV2(output_channel_5_3, output_channel_5_3, 1, 1),
                                       )
        self.up5_2to6_1 = nn.PixelShuffle(2)  # 940/4 = 240
        self.up5_3to6_1 = nn.PixelShuffle(4)  # 1920/16 = 120
        #     256 + 240 + 120 = 616

    def forward(self, x):
        x = self.conv1(x)
        s1 = self.maxpool(x)  # 1/4
        s2_1 = self.stage_2_1(s1)   # 1/8
        s2_1_to_3_2 = self.down2_1to3_2(s2_1)

        s3_1 = self.stage_3_1(s2_1)
        s3_2 = self.stage_3_2(s2_1_to_3_2)
        s3_1_to_4_2 = self.down3_1to4_2(s3_1)
        s3_2_to_4_1 = self.up3_2to4_1(s3_2)

        s4_1 = self.stage_4_1(_cat2(s3_1, s3_2_to_4_1))
        s4_2 = self.stage_4_2(_cat2(s3_1_to_4_2, s3_2))
        s4_1_to_5_2 = self.down4_1to5_2(s4_1)
        # s4_1_to_5_3 = self.down4_1to5_3(s4_1)
        s4_1_to_5_3 = self.down4_1to5_3(s4_1_to_5_2)  # 4_1--> 5_2--> 5_3
        s4_2_to_5_1 = self.up4_2to5_1(s4_2)
        s4_2_to_5_3 = self.down4_2to5_3(s4_2)

        s5_1 = self.stage_5_1(_cat2(s4_1, s4_2_to_5_1))   # 1/8
        s5_2 = self.stage_5_2(_cat2(s4_1_to_5_2, s4_2))
        # print(s4_1_to_5_3.size(), s4_2_to_5_3.size())
        s5_3 = self.stage_5_3(_cat2(s4_1_to_5_3, s4_2_to_5_3))
        # print(s5_1.size(), s5_2.size(), s5_3.size())

        y = _cat3(s5_1, self.up5_2to6_1(s5_2), self.up5_3to6_1(s5_3))  #
        return y


class LWShuffleNetV2_HRv2(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_HRv2, self).__init__()
        print('LWShuffleNetV2_HRv2, using InvertedResidualV2.')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            # self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
            self.stage_out_channels = [-1, 24, 64, 128, 192, 256, 1024]  # 2.9M, 2.57G, c=388  580
            # self.stage_out_channels = [-1, 24, 64, 128, 192, 288, 1024]  # 2.94M, 2.71G, c=420
            # self.stage_out_channels = [-1, 24, 64, 128, 192, 320, 1024]  # 2.98M, 2.86G, c=452
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        df = 1  # down-sample channel expansion factor
        input_channel_2_1 = self.stage_out_channels[1]
        output_channel_2_1 = self.stage_out_channels[2]
        self.stage_2_1 = nn.Sequential(InvertedResidualV2(input_channel_2_1, output_channel_2_1, 2, 2),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     )  # 1/8


        input_channel_3_1 = output_channel_2_1
        output_channel_3_1 = self.stage_out_channels[3]
        # print('3_1:{}, {}'.format(input_channel_3_1, output_channel_3_1))
        self.stage_3_1 = nn.Sequential(InvertedResidualV2(input_channel_3_1, output_channel_3_1, 1, 2),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     )

        input_channel_4_1 = self.stage_out_channels[3]  # + output_channel_3_2//4
        output_channel_4_1 = self.stage_out_channels[4]
        # print('4_1:{}, {}'.format(input_channel_4_1, output_channel_4_1))
        self.stage_4_1 = nn.Sequential(InvertedResidualV2(input_channel_4_1, output_channel_4_1, 1, 2),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     )

        input_channel_5_1 = output_channel_4_1  # + output_channel_4_2 // 4
        output_channel_5_1 = self.stage_out_channels[5]
        # print('5_1:{}, {}'.format(input_channel_5_1, output_channel_5_1))
        self.stage_5_1 = nn.Sequential(InvertedResidualV2(input_channel_5_1, output_channel_5_1, 1, 2),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     )


        # self.conv_cat2 = nn.Conv2d(output_channel_2_1, output_channel_2_1//2, 1, bias=True)
        # self.conv_cat3 = nn.Conv2d(output_channel_3_1, output_channel_3_1//2, 1, bias=True)
        # self.conv_cat4 = nn.Conv2d(output_channel_4_1, output_channel_4_1//2, 1, bias=True)
        #     256 + 240 + 120 = 616

    def forward(self, x):
        x = self.conv1(x)
        s1 = self.maxpool(x)  # 1/4
        s2_1 = self.stage_2_1(s1)   # 1/8
        s3_1 = self.stage_3_1(s2_1)
        s4_1 = self.stage_4_1(s3_1)
        s5_1 = self.stage_5_1(s4_1)   # 1/8
        # y = torch.cat((self.conv_cat2(s2_1), self.conv_cat3(s3_1), self.conv_cat4(s4_1), s5), dim=1)
        return s5_1



class LWShuffleNetV2_HR_catv2(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_HR_catv2, self).__init__()
        print('LWShuffleNetV2_HR_catv2, using InvertedResidualV2. 1/16, cat')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            # self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
            self.stage_out_channels = [-1, 24, 64, 128, 192, 256, 1024]  # 2.9M, 2.57G, c=388  580
            # self.stage_out_channels = [-1, 24, 64, 128, 192, 288, 1024]  # 2.94M, 2.71G, c=420
            # self.stage_out_channels = [-1, 24, 64, 128, 192, 320, 1024]  # 2.98M, 2.86G, c=452
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        df = 1  # down-sample channel expansion factor
        input_channel_2_1 = self.stage_out_channels[1]
        output_channel_2_1 = self.stage_out_channels[2]
        self.stage_2_1 = nn.Sequential(InvertedResidualV2(input_channel_2_1, output_channel_2_1, 2, 2),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     InvertedResidualV2(output_channel_2_1, output_channel_2_1, 1, 1),
                                     )  # 1/8
        self.down2_1to3_2 = nn.Sequential(nn.Conv2d(output_channel_2_1, output_channel_2_1 * df, 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channel_2_1 * df),
                                          nn.ReLU(inplace=True)
                                          )

        input_channel_3_1 = output_channel_2_1
        output_channel_3_1 = self.stage_out_channels[3]
        # print('3_1:{}, {}'.format(input_channel_3_1, output_channel_3_1))
        self.stage_3_1 = nn.Sequential(InvertedResidualV2(input_channel_3_1, output_channel_3_1, 1, 2),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     InvertedResidualV2(output_channel_3_1, output_channel_3_1, 1, 1),
                                     )
        self.down3_1to4_2 = nn.Sequential(nn.Conv2d(output_channel_3_1, output_channel_3_1 * df, 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channel_3_1 * df),
                                          nn.ReLU(inplace=True)
                                          )

        input_channel_3_2 = output_channel_2_1 * df
        output_channel_3_2 = input_channel_3_2
        # print('3_2:{}, {}'.format(input_channel_3_2, output_channel_3_2))
        self.stage_3_2 = nn.Sequential(InvertedResidualV2(input_channel_3_2, output_channel_3_2, 1, 2),
                                       InvertedResidualV2(output_channel_3_2, output_channel_3_2, 1, 1),
                                       InvertedResidualV2(output_channel_3_2, output_channel_3_2, 1, 1),
                                       # InvertedResidualV2(output_channel_3_2, output_channel_3_2, 1, 1),
                                       )
        self.up3_2to4_1 = nn.PixelShuffle(2)

        input_channel_4_1 = self.stage_out_channels[3] + output_channel_3_2//4
        output_channel_4_1 = self.stage_out_channels[4]
        # print('4_1:{}, {}'.format(input_channel_4_1, output_channel_4_1))
        self.stage_4_1 = nn.Sequential(InvertedResidualV2(input_channel_4_1, output_channel_4_1, 1, 2),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     InvertedResidualV2(output_channel_4_1, output_channel_4_1, 1, 1),
                                     )
        self.down4_1to5_2 = nn.Sequential(nn.Conv2d(output_channel_4_1, output_channel_4_1 * df, 3, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channel_4_1 * df),
                                          nn.ReLU(inplace=True)
                                          )
        # self.down4_1to5_3 = nn.Sequential(nn.Conv2d(output_channel_4_1, output_channel_4_1 * df, 3, 2, 1, bias=False),
        #                                   nn.Conv2d(output_channel_4_1 * df, output_channel_4_1 * (df + df), 3, 2, 1, bias=False),
        #                                   nn.BatchNorm2d(output_channel_4_1 * (df + df)),
        #                                   nn.ReLU(inplace=True)
        #                                   )

        self.down4_1to5_3 = nn.Sequential(
            # nn.Conv2d(output_channel_4_1, output_channel_4_1 * df, 3, 2, 1, bias=False),
            nn.Conv2d(output_channel_4_1 * df, output_channel_4_1 * (df + df), 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel_4_1 * (df + df)),
            nn.ReLU(inplace=True)
        )

        input_channel_4_2 = output_channel_3_2 + output_channel_3_1 * df
        output_channel_4_2 = input_channel_4_2
        # print('4_2:{}, {}'.format(input_channel_4_2, output_channel_4_2))
        self.stage_4_2 = nn.Sequential(InvertedResidualV2(input_channel_4_2, output_channel_4_2, 1, 2),
                                       InvertedResidualV2(output_channel_4_2, output_channel_4_2, 1, 1),
                                       InvertedResidualV2(output_channel_4_2, output_channel_4_2, 1, 1),
                                       # InvertedResidualV2(output_channel_4_2, output_channel_4_2, 1, 1),
                                       )
        self.up4_2to5_1 = nn.PixelShuffle(2)
        self.down4_2to5_3 = nn.Sequential(
            nn.Conv2d(output_channel_4_2, output_channel_4_2 * df, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel_4_2 * df),
            nn.ReLU(inplace=True)
            )

        input_channel_5_1 = output_channel_4_1 + output_channel_4_2 // 4
        output_channel_5_1 = self.stage_out_channels[5]
        # print('5_1:{}, {}'.format(input_channel_5_1, output_channel_5_1))
        self.stage_5_1 = nn.Sequential(InvertedResidualV2(input_channel_5_1, output_channel_5_1, 1, 2),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     InvertedResidualV2(output_channel_5_1, output_channel_5_1, 1, 1),
                                     )

        input_channel_5_2 = output_channel_4_2 + output_channel_4_1 * df
        output_channel_5_2 = input_channel_5_2
        # print('5_2:{}, {}'.format(input_channel_5_2, output_channel_5_2))
        self.stage_5_2 = nn.Sequential(InvertedResidualV2(input_channel_5_2, output_channel_5_2, 1, 2),
                                       InvertedResidualV2(output_channel_5_2, output_channel_5_2, 1, 1),
                                       InvertedResidualV2(output_channel_5_2, output_channel_5_2, 1, 1),
                                       # InvertedResidualV2(output_channel_5_2, output_channel_5_2, 1, 1),
                                       )

        input_channel_5_3 = output_channel_4_2 * df + output_channel_4_1 * (df + df)
        output_channel_5_3 = input_channel_5_3
        # print('5_3:{}, {}'.format(input_channel_5_3, output_channel_5_3))
        self.stage_5_3 = nn.Sequential(InvertedResidualV2(input_channel_5_3, output_channel_5_3, 1, 2),
                                       InvertedResidualV2(output_channel_5_3, output_channel_5_3, 1, 1),
                                       # InvertedResidualV2(output_channel_5_3, output_channel_5_3, 1, 1),
                                       # InvertedResidualV2(output_channel_5_3, output_channel_5_3, 1, 1),
                                       )
        self.up5_2to6_1 = nn.PixelShuffle(2)  # 940/4 = 240
        self.up5_3to6_1 = nn.PixelShuffle(4)  # 1920/16 = 120

        self.conv_cat2 = nn.Conv2d(output_channel_2_1, output_channel_2_1//2, 1, bias=True)
        self.conv_cat3 = nn.Conv2d(output_channel_3_1, output_channel_3_1//2, 1, bias=True)
        self.conv_cat4 = nn.Conv2d(output_channel_4_1, output_channel_4_1//2, 1, bias=True)
        #     256 + 240 + 120 = 616

    def forward(self, x):
        x = self.conv1(x)
        s1 = self.maxpool(x)  # 1/4
        s2_1 = self.stage_2_1(s1)   # 1/8
        s2_1_to_3_2 = self.down2_1to3_2(s2_1)

        s3_1 = self.stage_3_1(s2_1)
        s3_2 = self.stage_3_2(s2_1_to_3_2)
        s3_1_to_4_2 = self.down3_1to4_2(s3_1)
        s3_2_to_4_1 = self.up3_2to4_1(s3_2)

        s4_1 = self.stage_4_1(_cat2(s3_1, s3_2_to_4_1))
        s4_2 = self.stage_4_2(_cat2(s3_1_to_4_2, s3_2))
        s4_1_to_5_2 = self.down4_1to5_2(s4_1)
        # s4_1_to_5_3 = self.down4_1to5_3(s4_1)
        s4_1_to_5_3 = self.down4_1to5_3(s4_1_to_5_2)  # 4_1--> 5_2--> 5_3
        s4_2_to_5_1 = self.up4_2to5_1(s4_2)
        s4_2_to_5_3 = self.down4_2to5_3(s4_2)

        s5_1 = self.stage_5_1(_cat2(s4_1, s4_2_to_5_1))   # 1/8
        s5_2 = self.stage_5_2(_cat2(s4_1_to_5_2, s4_2))
        # print(s4_1_to_5_3.size(), s4_2_to_5_3.size())
        s5_3 = self.stage_5_3(_cat2(s4_1_to_5_3, s4_2_to_5_3))
        # print(s5_1.size(), s5_2.size(), s5_3.size())

        s5 = _cat3(s5_1, self.up5_2to6_1(s5_2), self.up5_3to6_1(s5_3))  #

        y = torch.cat((self.conv_cat2(s2_1), self.conv_cat3(s3_1), self.conv_cat4(s4_1), s5), dim=1)
        return y


class ShuffleNetV2_v1_cat(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_v1_cat, self).__init__()
        print('ShuffleNetV2_v1_cat, using InvertedResidualV2.')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidualV2(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidualV2(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        f1 = self.maxpool(x)  # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        y = torch.cat((f2, f3, f4), dim=1)  # 116, 232, 464
        return y


class ShuffleNetV2_v2(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_v2, self).__init__()
        print('ShuffleNetV2_v2, using InvertedResidualV2, replace maxpooling with conv.')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(8),
                                   # nn.ReLU(inplace=True)
                                   )

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential(nn.Conv2d(8, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   # nn.ReLU(inplace=True)
                                     )

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidualV2(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidualV2(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)  # 1/4
        x = self.stage_2(x)   # 1/8
        x = self.stage_3(x)   # 1/16
        x = self.stage_4(x)   # 1/32
        return x


class ShuffleNetV2_v3(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_v3, self).__init__()
        print('ShuffleNetV2_v3, using InvertedResidualV2, replace maxpooling with Resblock.')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(8),
                                   # nn.ReLU(inplace=True)
                                   )

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1, bias=True),
                                     nn.Conv2d(16, input_channel, 3, 2, 1, bias=False),
                                     nn.BatchNorm2d(input_channel),
                                     # nn.ReLU(inplace=True)
                                     )

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidualV2(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidualV2(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)  # 1/4
        x = self.stage_2(x)   # 1/8
        x = self.stage_3(x)   # 1/16
        x = self.stage_4(x)   # 1/32
        return x


# --- width_mult = 0.5 size: 1/8, 1/16, 1/32
# Params: 143136, 143.14K
# Flops:  201052800.0, 201.05M
# --- width_mult = 0.5 size: 1/8, 1/8, 1/8
# Params: 143136, 143.14K
# Flops:  759283200.0, 759.28M
# --- width_mult = 1.0, size: 1/8, 1/8, 1/8
# Params: 776420, 776.42K
# Flops:  3825158400.0, 3.83G
class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2, self).__init__()
        print('ShuffleNetV2')
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)  # 1/4
        x = self.stage_2(x)   # 1/8
        x = self.stage_3(x)   # 1/16
        x = self.stage_4(x)   # 1/32
        return x


# Params: 786989, 786.99K
# Flops:  2906917200.0, 2.91G
class ShuffleNetV2_cat(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_cat, self).__init__()
        print('ShuffleNetV2_cat width mult={}'.format(width_mult))
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        f1 = self.maxpool(x)  # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        y = torch.cat((f2, f3, f4), dim=1)  # 116, 232, 464

        return y


class ShuffleNetV2_Adaptive_cat(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_Adaptive_cat, self).__init__()
        print('ShuffleNetV2_Adaptive_cat width mult={}'.format(width_mult))
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        in_channel = self.stage_out_channels[2]
        self.pw_adapter2 = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid()
                                         )

        in_channel = self.stage_out_channels[3]
        self.pw_adapter3 = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid()
                                         )

        in_channel = self.stage_out_channels[4]
        self.pw_adapter4 = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid()
                                         )

    def forward(self, x):
        x = self.conv1(x)
        f1 = self.maxpool(x)  # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        mx2 = self.pw_adapter2(f2)
        mx3 = self.pw_adapter3(f3)
        mx4 = self.pw_adapter4(f4)

        # basis_channel = f2.size(1)
        mx_c2 = mx2.expand(-1, f2.size(1), -1, -1)  # (BS, basis_channel, H, W)
        mx_c3 = mx3.expand(-1, f3.size(1), -1, -1)  # (BS, basis_channel, H, W)
        mx_c4 = mx4.expand(-1, f4.size(1), -1, -1)  # (BS, basis_channel, H, W)

        f2 = f2 * mx_c2
        f3 = f3 * mx_c3
        f4 = f4 * mx_c4

        y = torch.cat((f2, f3, f4), dim=1)  # 116, 232, 464

        return y


class ShuffleNetV2_Adaptive_catV2(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_Adaptive_catV2, self).__init__()
        print('ShuffleNetV2_Adaptive_catV2 width mult={}'.format(width_mult))
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        in_channel = self.stage_out_channels[2]
        self.pw_adapter2 = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
                                         # nn.ReLU(inplace=True),
                                         nn.LeakyReLU(negative_slope=0.3, inplace=True)
                                         )

        in_channel = self.stage_out_channels[3]
        self.pw_adapter3 = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
                                         # nn.ReLU(inplace=True),
                                         nn.LeakyReLU(negative_slope=0.3, inplace=True)
                                         )

        in_channel = self.stage_out_channels[4]
        self.pw_adapter4 = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1, bias=False),
                                         # nn.ReLU(inplace=True),
                                         nn.LeakyReLU(negative_slope=0.3, inplace=True)
                                         )

    def forward(self, x):
        x = self.conv1(x)
        f1 = self.maxpool(x)  # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        mx2 = self.pw_adapter2(f2)
        mx3 = self.pw_adapter3(f3)
        mx4 = self.pw_adapter4(f4)

        mx2 = torch.clamp(mx2, min=0.01, max=2.0)
        mx3 = torch.clamp(mx3, min=0.01, max=2.0)
        mx4 = torch.clamp(mx4, min=0.01, max=2.0)

        # basis_channel = f2.size(1)
        mx_c2 = mx2.expand(-1, f2.size(1), -1, -1)  # (BS, basis_channel, H, W)
        mx_c3 = mx3.expand(-1, f3.size(1), -1, -1)  # (BS, basis_channel, H, W)
        mx_c4 = mx4.expand(-1, f4.size(1), -1, -1)  # (BS, basis_channel, H, W)

        f2 = f2 * mx_c2
        f3 = f3 * mx_c3
        f4 = f4 * mx_c4

        y = torch.cat((f2, f3, f4), dim=1)  # 116, 232, 464

        return y


# def mix_adapter(x, mx, basis_channel):
#     # mx = self.conv_mx(x)
#     # mx = self.softmax(mx)  # (BS, 1, D, H, W)
#
#     # mx_c = torch.unsqueeze(mx, dim=2)  # (BS, 1,  D, H, W)
#     mx_c = mx.expand(-1, basis_channel, -1, -1, -1)  # (BS, basis_channel, D, H, W)
#
#     mx_tuple = torch.split(mx_c, 1, dim=1)  # basis_num x (BS, 1, basis_channel, D, H, W)
#     # print("Length of mx_tuple:{}".format(len(mx_tuple)))
#
#     # mxs = (torch.squeeze(mxi, dim=1) for mxi in mx_tuple)
#     mxs = map(lambda v: torch.squeeze(v, dim=1), mx_tuple)
#     # print("Length of mxs:{}".format(len(mxs)))
#     return mxs


class ShuffleNetV2_add(nn.Module):
    def __init__(self, width_mult=0.5):
        super(ShuffleNetV2_add, self).__init__()
        print('ShuffleNetV2_add width mult={}'.format(width_mult))
        # assert (input_size[0] % 32 == 0 and input_size[1] % 32 == 0)
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    stride = 2 if idxstage == 0 else 1
                    # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

    def forward(self, x):
        x = self.conv1(x)
        f1 = self.maxpool(x)  # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        # y = torch.cat((f2, f3, f4), dim=1)  # 116, 232, 464

        c2 = f2.size(1)
        c3 = f3.size(1)
        # c4 = f4.size(1)

        y1 = f4[:, 0:c2, ] + f3[:, 0:c2, ] + f2
        y2 = f4[:, c2:c3, ] + f3[:, c2:c3, ]
        y3 = f4[:, c3:, ]
        y = torch.cat((y1, y2, y3), dim=1)

        return y


# width_mult = 0.5, ASPP
# Params: 1489384, 1.49M
# Flops:  3033667200.0, 3.03G
# width_mult = 0.5, LWASPP
# Params: 1286952, 1.29M
# Flops:  2836137600.0, 2.84G
# light-weight decoder
# Params: 449984, 449.98K
# Flops:  753259200.0, 753.26M
# width_mult = 1.0, LWASPP
# Params: 1106473, 1.11M
# Flops:  1353631200.0, 1.35G
class LWShuffleNetV2(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2, self).__init__()
        print('LWShuffleNetV2, width_mult={}'.format(width_mult))
        # width_mult = 0.5
        # width_mult = 1.0
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        # Multi-stage ASPP
        # TODO funetune the c_out
        # self.aspp2 = ASPP(c_in=self.stage_out_channels[2], c_out=24, dilation=[1, 2, 4, 8, 12], global_pool=False)
        # self.aspp3 = ASPP(c_in=self.stage_out_channels[3], c_out=64, dilation=[1, 2, 4, 8], global_pool=False)
        # self.aspp4 = ASPP(c_in=self.stage_out_channels[4], c_out=128, dilation=[1, 2, 4], global_pool=False)
        # 0.5: 48, 96, 192
        # 1.0: 116, 232, 464
        self.aspp2 = LWASPP(c_in=self.stage_out_channels[2], c=24, c_out=24, dilation=[1, 2, 4, 8, 12], global_pool=False)
        self.aspp3 = LWASPP(c_in=self.stage_out_channels[3], c=48, c_out=48, dilation=[1, 2, 4, 8], global_pool=False)
        self.aspp4 = LWASPP(c_in=self.stage_out_channels[4], c=48, c_out=48, dilation=[1, 2, 4], global_pool=False)

        self.conv_aspp2 = nn.Sequential(nn.Conv2d(120, 64, 1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU())

        self.conv_aspp3 = nn.Sequential(nn.Conv2d(192, 256, 1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
        # 128*3=384
        # 64*3=192
        # 48*3 = 144
        self.conv_aspp4 = nn.Sequential(nn.Conv2d(144, 384, 1, bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU())

        # Upsample
        # self.upsample_aspp2 = nn.PixelShuffle(1)  # 64
        self.upsample_aspp3 = nn.PixelShuffle(2)  # 256 --> 64*2*2
        self.upsample_aspp4 = nn.PixelShuffle(4)  # 384 --> 24*4*4

        # Decoder
        # 64 + 64 + 24 = 152
        _c = 152
        self.decoder_conv1 = nn.Sequential(nn.Conv2d(_c, _c, 1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())
        # self.upsample = nn.PixelShuffle(2)  # 152 --> 38*2*2

        # _c = 38
        _c = 152
        # self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=3, stride=1, padding=1, bias=True),
        #                                    nn.BatchNorm2d(_c),
        #                                    nn.ReLU())

        self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, stride=1, padding=0, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())
        # _c = 38
        # _c = 152
        # self.decoder_conv3 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=3, stride=1, padding=1, bias=True),
        #                                    nn.BatchNorm2d(_c),
        #                                    nn.ReLU())
        # self.decoder_conv3 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, stride=1, padding=0, bias=True),
        #                                    nn.BatchNorm2d(_c),
        #                                    nn.ReLU())
        # self.output = nn.Conv2d(_c, n_classes, 1)

    def forward(self, x):
        bs, c, h, w = x.size()
        # padding
        # if h == 720:  # 720*1280 -->736*1280
        # if h % 32 != 0:
        #     m, n = divmod(h, 32)
        #     ph = int(((m+1)*32-h)/2)
        #     x = F.pad(x, (0, 0, ph, ph), "constant", 0)
        # print(x.size())

        x = self.conv1(x)       # 1/2
        f1 = self.maxpool(x)    # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        x_f2 = self.aspp2(f2)
        x_f2 = self.conv_aspp2(x_f2)    # 1/8
        # print('x_f2.size :', x_f2.size())  # [1, 64, 60, 80]

        x_f3 = self.aspp3(f3)
        x_f3 = self.conv_aspp3(x_f3)        # 1/16
        # print('x_f3.size :', x_f3.size())  # [1, 256, 30, 40]
        x_f3 = self.upsample_aspp3(x_f3)    # 1/8  [1, 64, 60, 80]

        x_f4 = self.aspp4(f4)
        x_f4 = self.conv_aspp4(x_f4)        # 1/32
        # print('x_f4.size :', x_f4.size())  # [1, 384, 15, 20]
        x_f4 = self.upsample_aspp4(x_f4)    # 1/8 [1, 24, 60, 80]

        # crop the feature map to the same size as the x_f2 after upsamling
        # 360//32
        # print('After upsample:', x_f2.size(), x_f3.size(), x_f4.size())

        # h
        h_min = min(x_f2.size(2), x_f3.size(2), x_f4.size(2))
        ds2 = x_f2.size(2) - h_min
        ds3 = x_f3.size(2) - h_min
        ds4 = x_f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = x_f2[:, :, ds2//2:x_f2.size(2)-(ds2-ds2//2), :]
        x_f3 = x_f3[:, :, ds3//2:x_f3.size(2)-(ds3-ds3//2), :]
        x_f4 = x_f4[:, :, ds4//2:x_f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        x_concat = torch.cat((x_f2, x_f3, x_f4), dim=1)
        y = self.decoder_conv1(x_concat)

        # print('y.size 1:', y.size())

        # y = self.upsample(y)

        y = self.decoder_conv2(y)
        # y = self.decoder_conv3(y)

        # print('y.size 2:', y.size())

        # y = F.interpolate(y, scale_factor=4, mode='bilinear', align_corners=True)

        # y = self.output(y)

        # print('y.size 3:', y.size())

        # y = torch.sigmoid(y)  # In multi-channel binary segmentation, the value of each channel must between[0, 1]
        # if h % 32 != 0:
        #     y = y[:, :, ph:h+ph, :]
        return y


# increase the number of channels
class LWShuffleNetV2_MultiASPP192(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_MultiASPP192, self).__init__()
        print('LWShuffleNetV2_MultiASPP192, width_mult={}'.format(width_mult))
        # width_mult = 0.5
        # width_mult = 1.0
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        # Multi-stage ASPP
        # TODO funetune the c_out
        # 0.5: 48, 96, 192
        # 1.0: 116, 232, 464
        self.aspp2 = LWASPP(c_in=self.stage_out_channels[2], c=48, c_out=48, dilation=[1, 2, 4, 8, 12], global_pool=False)
        self.aspp3 = LWASPP(c_in=self.stage_out_channels[3], c=48, c_out=48, dilation=[1, 2, 4, 8], global_pool=False)
        self.aspp4 = LWASPP(c_in=self.stage_out_channels[4], c=48, c_out=48, dilation=[1, 2, 4], global_pool=False)

        # 48 * 5 = 240
        self.conv_aspp2 = nn.Sequential(nn.Conv2d(240, 104, 1, bias=True),
                                        nn.BatchNorm2d(104),
                                        nn.ReLU())

        self.conv_aspp3 = nn.Sequential(nn.Conv2d(192, 256, 1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())
        # 128*3=384
        # 64*3=192
        # 48*3 = 144
        self.conv_aspp4 = nn.Sequential(nn.Conv2d(144, 384, 1, bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU())

        # Upsample
        # self.upsample_aspp2 = nn.PixelShuffle(1)  # 64
        self.upsample_aspp3 = nn.PixelShuffle(2)  # 256 --> 64*2*2
        self.upsample_aspp4 = nn.PixelShuffle(4)  # 384 --> 24*4*4

        # Decoder
        # _c = 152  # 64 + 64 + 24 = 152
        _c = 192  # 104 + 64 + 24 = 192
        self.decoder_conv1 = nn.Sequential(nn.Conv2d(_c, _c, 1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

        # self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, stride=1, padding=0, bias=True),
        #                                    nn.BatchNorm2d(_c),
        #                                    nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)       # 1/2
        f1 = self.maxpool(x)    # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        x_f2 = self.aspp2(f2)
        x_f2 = self.conv_aspp2(x_f2)    # 1/8
        # print('x_f2.size :', x_f2.size())  # [1, 64, 60, 80]

        x_f3 = self.aspp3(f3)
        x_f3 = self.conv_aspp3(x_f3)        # 1/16
        # print('x_f3.size :', x_f3.size())  # [1, 256, 30, 40]
        x_f3 = self.upsample_aspp3(x_f3)    # 1/8  [1, 64, 60, 80]

        x_f4 = self.aspp4(f4)
        x_f4 = self.conv_aspp4(x_f4)        # 1/32
        # print('x_f4.size :', x_f4.size())  # [1, 384, 15, 20]
        x_f4 = self.upsample_aspp4(x_f4)    # 1/8 [1, 24, 60, 80]

        # crop the feature map to the same size as the x_f2 after upsamling
        # 360//32
        # print('After upsample:', x_f2.size(), x_f3.size(), x_f4.size())

        # h
        h_min = min(x_f2.size(2), x_f3.size(2), x_f4.size(2))
        ds2 = x_f2.size(2) - h_min
        ds3 = x_f3.size(2) - h_min
        ds4 = x_f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = x_f2[:, :, ds2//2:x_f2.size(2)-(ds2-ds2//2), :]
        x_f3 = x_f3[:, :, ds3//2:x_f3.size(2)-(ds3-ds3//2), :]
        x_f4 = x_f4[:, :, ds4//2:x_f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        x_concat = torch.cat((x_f2, x_f3, x_f4), dim=1)
        y = self.decoder_conv1(x_concat)
        # y = self.decoder_conv2(y)

        return y


# increase the number of channels
class LWShuffleNetV2_MultiASPP(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_MultiASPP, self).__init__()
        print('LWShuffleNetV2_MultiASPP, width_mult={}'.format(width_mult))
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        # Multi-stage ASPP
        # TODO funetune the c_out
        # 0.5:  48,  96, 192
        # 1.0: 116, 232, 464
        c_in2, c_out2 = self.stage_out_channels[2], self.stage_out_channels[2] // 2
        c_in3, c_out3 = self.stage_out_channels[3], self.stage_out_channels[3] // 2
        c_in4, c_out4 = self.stage_out_channels[4], self.stage_out_channels[4] // 2
        self.conv_aspp_in2 = nn.Sequential(nn.Conv2d(c_in2, c_out2, 1, bias=True), nn.BatchNorm2d(c_out2), nn.ReLU())
        self.conv_aspp_in3 = nn.Sequential(nn.Conv2d(c_in3, c_out3, 1, bias=True), nn.BatchNorm2d(c_out3), nn.ReLU())
        self.conv_aspp_in4 = nn.Sequential(nn.Conv2d(c_in4, c_out4, 1, bias=True), nn.BatchNorm2d(c_out4), nn.ReLU())

        # 0.5: ( 48,  96, 192)//2
        # 1.0: (116, 232, 464)//2
        self.aspp2 = LWASPP(c_in=c_out2, c=c_out2//2, c_out=c_out2, dilation=[1, 2, 4, 8, 12], global_pool=False)
        self.aspp3 = LWASPP(c_in=c_out3, c=c_out3//2, c_out=c_out3, dilation=[1, 2, 4, 8], global_pool=False)
        self.aspp4 = LWASPP(c_in=c_out4, c=c_out4//2, c_out=c_out4, dilation=[1, 2, 4, 8], global_pool=False)

        # 0.5: ( 48*5,  96*4, 192*4)//4
        # 1.0: (116*5, 232*4, 464*4)//4
        c2 = c_out2 * 5
        c3 = c_out3 * 4
        c4 = c_out4 * 4
        self.conv_aspp_out2 = nn.Sequential(nn.Conv2d(c2, c2//2, 1, bias=True), nn.BatchNorm2d(c2//2), nn.ReLU())
        self.conv_aspp_out3 = nn.Sequential(nn.Conv2d(c3, c3//2, 1, bias=True), nn.BatchNorm2d(c3//2), nn.ReLU())
        self.conv_aspp_out4 = nn.Sequential(nn.Conv2d(c4, c4//2, 1, bias=True), nn.BatchNorm2d(c4//2), nn.ReLU())

        # Upsample
        # self.upsample_aspp2 = nn.PixelShuffle(1)
        self.upsample_aspp3 = nn.PixelShuffle(2)
        self.upsample_aspp4 = nn.PixelShuffle(4)

        # Decoder
        # 0.5: ( 48*5,  96*4//4, 192*4//16)//2 = 120,  48, 24 = 192
        # 1.0: (116*5, 232*4//4, 464*4//16)//2 = 290, 116, 58 = 464

        # 0.5: ( 48*5,  96*4//4, 192*4//16)//4 = 60,  24, 12 = 96
        # 1.0: (116*5, 232*4//4, 464*4//16)//4 = 145, 58, 29 = 232
        _c = c2//2 + c3//2 // 4 + c4//2 // 16  #

        self.decoder_conv1 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

        self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)       # 1/2
        f1 = self.maxpool(x)    # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        x_f2 = self.conv_aspp_in2(f2)
        x_f2 = self.aspp2(x_f2)
        x_f2 = self.conv_aspp_out2(x_f2)    # 1/8
        # print('x_f2.size :', x_f2.size())  # [1, 64, 60, 80]

        x_f3 = self.conv_aspp_in3(f3)
        x_f3 = self.aspp3(x_f3)
        x_f3 = self.conv_aspp_out3(x_f3)        # 1/16
        # print('x_f3.size :', x_f3.size())  # [1, 256, 30, 40]
        x_f3 = self.upsample_aspp3(x_f3)    # 1/8  [1, 64, 60, 80]

        x_f4 = self.conv_aspp_in4(f4)
        x_f4 = self.aspp4(x_f4)
        x_f4 = self.conv_aspp_out4(x_f4)        # 1/32
        # print('x_f4.size :', x_f4.size())  # [1, 384, 15, 20]
        x_f4 = self.upsample_aspp4(x_f4)    # 1/8 [1, 24, 60, 80]

        # crop the feature map to the same size as the x_f2 after upsamling
        # 360//32
        # print('After upsample:', x_f2.size(), x_f3.size(), x_f4.size())

        # h
        h_min = min(x_f2.size(2), x_f3.size(2), x_f4.size(2))
        ds2 = x_f2.size(2) - h_min
        ds3 = x_f3.size(2) - h_min
        ds4 = x_f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = x_f2[:, :, ds2//2:x_f2.size(2)-(ds2-ds2//2), :]
        x_f3 = x_f3[:, :, ds3//2:x_f3.size(2)-(ds3-ds3//2), :]
        x_f4 = x_f4[:, :, ds4//2:x_f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        x_concat = torch.cat((x_f2, x_f3, x_f4), dim=1)
        y = self.decoder_conv1(x_concat)
        y = self.decoder_conv2(y)

        return y


class LWShuffleNetV2_single_ASPP(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_single_ASPP, self).__init__()
        print('LWShuffleNetV2_single_ASPP, width_mult={}'.format(width_mult))
        # width_mult = 0.5
        # width_mult = 1.0
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        # Multi-stage Feature Aggregation
        # Upsample
        # self.upsample_s2 = nn.PixelShuffle(1)  # 64
        self.upsample_s3 = nn.PixelShuffle(2)  # 256 = 64*2*2, 232 = 58*2*2
        self.upsample_s4 = nn.PixelShuffle(4)  # 384 = 24*4*4, 464 = 29*4*4

        # concatenate
        # 0.5: 48, 96, 192,
        # 1.0: 116, 232, 464
        # 1.5: 176, 352, 704
        if width_mult == 0.5:
            _c_cat_in = 84  # 48, 96/2/2=24, 192/4/4=12,  48+24+12=84
        elif width_mult == 1.0:
            _c_cat_in = 203  # 116 + 58 + 29 = 203
        elif width_mult == 1.5:
            _c_cat_in = 308  # 176, 352/2/2=88, 704/4/4=44, 176+88+44=308

        # _c_cat_out = 96
        # _c_aspp_in = 96
        # _c_aspp = 48
        # _c_aspp_out = 48

        _c_cat_out = _c_cat_in//2
        self.conv_cat = nn.Sequential(nn.Conv2d(_c_cat_in, _c_cat_out, 1, bias=True),
                                      nn.BatchNorm2d(_c_cat_out),
                                      nn.ReLU())

        # 0.5:  42, 21,  42, 168
        # 1.0: 101, 50, 101, 404
        _c_aspp_in, _c_aspp, _c_aspp_out = _c_cat_out, _c_cat_out//2, _c_cat_out
        self.aspp = LWASPP(c_in=_c_aspp_in, c=_c_aspp, c_out=_c_aspp_out, dilation=[1, 2, 4, 8], global_pool=False)

        # Decoder
        _c = _c_aspp_out * 4
        self.decoder_conv1 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

        self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)       # 1/2
        f1 = self.maxpool(x)    # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        # x_f2 = self.upsample_s2(f2)    # 1/8
        f3 = self.upsample_s3(f3)    # 1/8  [1, 64, 60, 80]
        f4 = self.upsample_s4(f4)    # 1/8 [1, 24, 60, 80]

        # crop the feature map to the same size as the x_f2 after upsamling
        # 360//32
        # print('After upsample:', x_f2.size(), x_f3.size(), x_f4.size())
        # h
        h_min = min(f2.size(2), f3.size(2), f4.size(2))
        ds2 = f2.size(2) - h_min
        ds3 = f3.size(2) - h_min
        ds4 = f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = f2[:, :, ds2//2:f2.size(2)-(ds2-ds2//2), :]
        x_f3 = f3[:, :, ds3//2:f3.size(2)-(ds3-ds3//2), :]
        x_f4 = f4[:, :, ds4//2:f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        y = torch.cat((x_f2, x_f3, x_f4), dim=1)
        y = self.conv_cat(y)
        y = self.aspp(y)
        y = self.decoder_conv1(y)
        y = self.decoder_conv2(y)
        return y



# Params: 1491373, 1.49M
# Flops:  3043214400.0, 3.04G
# LWASPP
# Params: 1288941, 1.29M
# Flops:  2845684800.0, 2.85G
# light-weight decoder
# Params: 451973, 451.97K
# Flops:  762806400.0, 762.81M
class LWNetwork(nn.Module):
    def __init__(self, width_mult=1.0, numkeypoints=18, numlims=19, multistage=0, backbone=None, head=None):  # Mod by Jie
        super(LWNetwork, self).__init__()
        if backbone == 'LWShuffleNetV2_baseline':
            self.network = ShuffleNetV2(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 464
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_HR_cat':
            self.network = LWShuffleNetV2_HR_cat(width_mult=width_mult)
            if width_mult == 1.0:
                # c_pose_in = 616
                c_pose_in = 412
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_HR_catv2':
            self.network = LWShuffleNetV2_HR_catv2(width_mult=width_mult)
            if width_mult == 1.0:
                # c_pose_in = 616
                # c_pose_in = 388
                # c_pose_in = 420
                # c_pose_in = 452
                c_pose_in = 580
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_HRv2':
            self.network = LWShuffleNetV2_HRv2(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 256
        elif backbone == 'LWShuffleNetV2_baseline_v1':
            self.network = ShuffleNetV2_v1(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 464
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_baseline_v1_cat':
            self.network = ShuffleNetV2_v1_cat(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 812
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_v1_16_cat':
            self.network = LWShuffleNetV2_v1_cat(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 464
            # elif width_mult == 0.5:
            #     c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_baseline_v2':
            self.network = ShuffleNetV2_v2(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 464
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_baseline_v3':
            self.network = ShuffleNetV2_v3(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 464
            elif width_mult == 0.5:
                c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_SingleASPP':
            self.network = LWShuffleNetV2_single_ASPP(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 404
            elif width_mult == 0.5:
                c_pose_in = 168
        elif backbone == 'LWShuffleNetV2_MultiASPP':
            self.network = LWShuffleNetV2_MultiASPP(width_mult=width_mult)
            if width_mult == 1.0:
                # c_pose_in = 464
                c_pose_in = 232
            elif width_mult == 0.5:
                # c_pose_in = 192
                c_pose_in = 96
        elif backbone == 'LWShuffleNetV2_mscat':
            self.network = LWShuffleNetV2_mscat(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 203
        elif backbone == 'LWShuffleNetV2_msadd':
            self.network = LWShuffleNetV2_msadd(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 116
        elif backbone == 'ShuffleNetV2_cat':
            self.network = ShuffleNetV2_cat(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 812
        elif backbone == 'ShuffleNetV2_Adaptive_cat':
            self.network = ShuffleNetV2_Adaptive_cat(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 812
        elif backbone == 'ShuffleNetV2_Adaptive_catV2':
            self.network = ShuffleNetV2_Adaptive_catV2(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 812
        elif backbone == 'ShuffleNetV2_add':
            self.network = ShuffleNetV2_add(width_mult=width_mult)
            if width_mult == 1.0:
                c_pose_in = 464
        elif backbone == 'LWShuffleNetV2_MultiASPP152':
            self.network = LWShuffleNetV2(width_mult=width_mult)
            c_pose_in = 152
        elif backbone == 'LWShuffleNetV2_SingleASPP192':
            self.network = LWShuffleNetV2_single_ASPP(width_mult=width_mult)
            c_pose_in = 192
        elif backbone == 'LWShuffleNetV2_MultiASPP192':
            self.network = LWShuffleNetV2_MultiASPP192(width_mult=width_mult)
            c_pose_in = 192
        else:
            print('Please set the right backbone.')
            exit(0)

        # head = 'ResidualAdapter'
        if head == 'ResidualAdapter':
            print("Network head == 'ResidualAdapter'")
            self.paf = nn.Sequential(
                ResidualAdapter(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in, kernel=3, residual=True),
                ResidualAdapter(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in//2, kernel=3, residual=False),
                nn.Conv2d(c_pose_in//2, numlims * 2, 1)
            )

            self.heatmap = nn.Sequential(
                ResidualAdapter(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in, kernel=3, residual=True),
                ResidualAdapter(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in//2, kernel=3, residual=False),
                nn.Conv2d(c_pose_in//2, numkeypoints + 1, 1)
            )
        if head == 'ResidualAdapterV2':
            print("Network head == 'ResidualAdapterV2'")
            # remove relu
            k = 1
            self.paf = nn.Sequential(
                ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in, kernel=k, residual=True),
                ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in//2, kernel=k, residual=False),
                nn.Conv2d(c_pose_in//2, numlims * 2, 1)
            )

            self.heatmap = nn.Sequential(
                ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in, kernel=k, residual=True),
                ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in//2, kernel=k, residual=False),
                nn.Conv2d(c_pose_in//2, numkeypoints + 1, 1)
            )
        if head == 'LWResidualAdapterV2':
            print("Network head == 'LWResidualAdapterV2'")
            # remove relu
            k = 1
            c = 96
            self.paf = nn.Sequential(
                # ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in, kernel=k, residual=True),
                # ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//8, c_out=c_pose_in//4, kernel=k, residual=False),
                ResidualAdapterV3(c_in=c_pose_in, c=c, c_out=(numlims * 2), kernel=k, groups=4, residual=False),
                # nn.Conv2d(c_pose_in//4, numlims * 2, 1)
            )

            self.heatmap = nn.Sequential(
                # ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//4, c_out=c_pose_in, kernel=k, residual=True),
                # ResidualAdapterV2(c_in=c_pose_in, c=c_pose_in//8, c_out=c_pose_in//4, kernel=k, residual=False),
                ResidualAdapterV3(c_in=c_pose_in, c=c, c_out=(numkeypoints + 1), kernel=k, groups=4, residual=False),
                # nn.Conv2d(c_pose_in//4, numkeypoints + 1, 1)
            )
        elif head == 'ResidualAdapterMix':
            print("Network head == 'ResidualAdapterMix'")
            k = 1
            self.paf = nn.Sequential(
                ResidualAdapterMix(c_in=c_pose_in, c=c_pose_in // 4, c_out=c_pose_in, kernel=k, residual=True),
                ResidualAdapterMix(c_in=c_pose_in, c=c_pose_in // 4, c_out=c_pose_in // 2, kernel=k, residual=False),
                nn.Conv2d(c_pose_in // 2, numlims * 2, 1)
            )

            self.heatmap = nn.Sequential(
                ResidualAdapterMix(c_in=c_pose_in, c=c_pose_in // 4, c_out=c_pose_in, kernel=k, residual=True),
                ResidualAdapterMix(c_in=c_pose_in, c=c_pose_in // 4, c_out=c_pose_in // 2, kernel=k, residual=False),
                nn.Conv2d(c_pose_in // 2, numkeypoints + 1, 1)
            )
        else:
            self.paf = nn.Conv2d(c_pose_in, numlims * 2, 1)
            self.heatmap = nn.Conv2d(c_pose_in, numkeypoints + 1, 1)  # channels: background + numkeypoints

        # self.th_paf = nn.Tanh()
        # self.sf_heatmap = nn.Softmax(dim=1)
        self.multistage = multistage
        # -----------
        if self.multistage > 0:
            num_channels = 128
            self.refinement_stages = nn.ModuleList()
            ch = numlims * 2 + numkeypoints + 1
            for i in range(self.multistage):
                # self.refinement_stages.append(nn.Conv2d(in_channels=ic1 + ch, out_channels=ch, kernel_size=3, padding=1))
                self.refinement_stages.append(RefinementStage(in_channels=c_pose_in + ch,
                                                              out_channels=num_channels,
                                                              num_heatmaps=numkeypoints + 1,
                                                              num_pafs=numlims * 2))

        for name, m in self.named_modules():
            if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network(x)
        # print(x.size())
        PAF = self.paf(x)
        HEAT = self.heatmap(x)
        # PAF = self.th_paf(PAF)
        # HEAT = self.sf_heatmap(HEAT)   # TODO

        stages_output = [PAF, HEAT]
        if self.multistage > 0:
            # stages_output=[]
            for refinement_stage in self.refinement_stages:
                # stages_output.append( refinement_stage(torch.cat([x, stages_output[-2], stages_output[-1]], dim=1)) )
                stages_output.extend( refinement_stage(torch.cat([x, stages_output[-2], stages_output[-1]], dim=1)) )

        return stages_output





# ------------------------------------------------------------------


class LWShuffleNetV2_mscat(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_mscat, self).__init__()
        print('LWShuffleNetV2_mscat, multi-stage feature concatenation, width_mult={}'.format(width_mult))
        # width_mult = 0.5
        # width_mult = 1.0
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        # Multi-stage Feature Aggregation
        # Upsample
        # self.upsample_s2 = nn.PixelShuffle(1)  # 64
        self.upsample_s3 = nn.PixelShuffle(2)  # 256 = 64*2*2, 232 = 58*2*2
        self.upsample_s4 = nn.PixelShuffle(4)  # 384 = 24*4*4, 464 = 29*4*4

        # concatenate
        # 0.5: 48, 96, 192,
        # 1.0: 116, 232, 464
        # 1.5: 176, 352, 704
        if width_mult == 0.5:
            _c_cat_in = 84  # 48, 96/2/2=24, 192/4/4=12,  48+24+12=84
        elif width_mult == 1.0:
            _c_cat_in = 203  # 116 + 58 + 29 = 203
        elif width_mult == 1.5:
            _c_cat_in = 308  # 176, 352/2/2=88, 704/4/4=44, 176+88+44=308

        # _c_cat_out = 96
        # _c_aspp_in = 96
        # _c_aspp = 48
        # _c_aspp_out = 48

        # _c_cat_out = _c_cat_in//2
        # self.conv_cat = nn.Sequential(nn.Conv2d(_c_cat_in, _c_cat_out, 1, bias=True),
        #                               nn.BatchNorm2d(_c_cat_out),
        #                               nn.ReLU())

        # 0.5:  42, 21,  42, 168
        # 1.0: 101, 50, 101, 404
        # _c_aspp_in, _c_aspp, _c_aspp_out = _c_cat_out, _c_cat_out//2, _c_cat_out
        # self.aspp = LWASPP(c_in=_c_aspp_in, c=_c_aspp, c_out=_c_aspp_out, dilation=[1, 2, 4, 8], global_pool=False)

        # Decoder
        _c = _c_cat_in
        self.decoder_conv1 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

        self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)       # 1/2
        f1 = self.maxpool(x)    # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        # x_f2 = self.upsample_s2(f2)    # 1/8
        f3 = self.upsample_s3(f3)    # 1/8  [1, 64, 60, 80]
        f4 = self.upsample_s4(f4)    # 1/8 [1, 24, 60, 80]

        # crop the feature map to the same size as the x_f2 after upsamling
        # 360//32
        # print('After upsample:', x_f2.size(), x_f3.size(), x_f4.size())
        # h
        h_min = min(f2.size(2), f3.size(2), f4.size(2))
        ds2 = f2.size(2) - h_min
        ds3 = f3.size(2) - h_min
        ds4 = f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = f2[:, :, ds2//2:f2.size(2)-(ds2-ds2//2), :]
        x_f3 = f3[:, :, ds3//2:f3.size(2)-(ds3-ds3//2), :]
        x_f4 = f4[:, :, ds4//2:f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        y = torch.cat((x_f2, x_f3, x_f4), dim=1)
        # y = self.conv_cat(y)
        # y = self.aspp(y)
        y = self.decoder_conv1(y)
        y = self.decoder_conv2(y)
        return y



class LWShuffleNetV2_msadd(nn.Module):
    def __init__(self, width_mult=1.0):
        super(LWShuffleNetV2_msadd, self).__init__()
        print('LWShuffleNetV2_msadd, multi-stage asymmetric feature addition, width_mult={}'.format(width_mult))
        # width_mult = 0.5
        # width_mult = 1.0
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []

        self.stage_2 = []
        self.stage_3 = []
        self.stage_4 = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # make it nn.Sequential
        self.stage_2 = nn.Sequential(self.features[0], self.features[1], self.features[2], self.features[3])
        self.stage_3 = nn.Sequential(self.features[4], self.features[5], self.features[6], self.features[7],
                                     self.features[8], self.features[9], self.features[10], self.features[11])
        self.stage_4 = nn.Sequential(self.features[12], self.features[13], self.features[14], self.features[15])

        # Multi-stage Feature Aggregation
        # Upsample
        # self.upsample_s2 = nn.PixelShuffle(1)  # 64
        self.upsample_s3 = nn.PixelShuffle(2)  # 256 = 64*2*2, 232 = 58*2*2
        self.upsample_s4 = nn.PixelShuffle(4)  # 384 = 24*4*4, 464 = 29*4*4

        # concatenate
        # 0.5: 48, 96, 192,
        # 1.0: 116, 232, 464
        # 1.5: 176, 352, 704
        if width_mult == 0.5:
            # _c_cat_in = 84  # 48, 96/2/2=24, 192/4/4=12,  48+24+12=84
            _c_cat_in = 48
        elif width_mult == 1.0:
            # _c_cat_in = 203  # 116 + 58 + 29 = 203
            _c_cat_in = 116
        elif width_mult == 1.5:
            # _c_cat_in = 308  # 176, 352/2/2=88, 704/4/4=44, 176+88+44=308
            _c_cat_in = 176

        # _c_cat_out = 96
        # _c_aspp_in = 96
        # _c_aspp = 48
        # _c_aspp_out = 48

        # _c_cat_out = _c_cat_in//2
        # self.conv_cat = nn.Sequential(nn.Conv2d(_c_cat_in, _c_cat_out, 1, bias=True),
        #                               nn.BatchNorm2d(_c_cat_out),
        #                               nn.ReLU())

        # 0.5:  42, 21,  42, 168
        # 1.0: 101, 50, 101, 404
        # _c_aspp_in, _c_aspp, _c_aspp_out = _c_cat_out, _c_cat_out//2, _c_cat_out
        # self.aspp = LWASPP(c_in=_c_aspp_in, c=_c_aspp, c_out=_c_aspp_out, dilation=[1, 2, 4, 8], global_pool=False)

        # Decoder
        _c = _c_cat_in
        self.decoder_conv1 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

        self.decoder_conv2 = nn.Sequential(nn.Conv2d(_c, _c, kernel_size=1, bias=True),
                                           nn.BatchNorm2d(_c),
                                           nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)       # 1/2
        f1 = self.maxpool(x)    # 1/4
        f2 = self.stage_2(f1)   # 1/8
        f3 = self.stage_3(f2)   # 1/16
        f4 = self.stage_4(f3)   # 1/32
        # print('stage_1:', f1.size(), 'stage_2:', f2.size(),
        #       'stage_3:', f3.size(), 'stage_4:', f4.size())

        # x_f2 = self.upsample_s2(f2)    # 1/8
        f3 = self.upsample_s3(f3)    # 1/8  [1, 64, 60, 80]
        f4 = self.upsample_s4(f4)    # 1/8 [1, 24, 60, 80]

        # crop the feature map to the same size as the x_f2 after upsamling
        # 360//32
        # print('After upsample:', x_f2.size(), x_f3.size(), x_f4.size())
        # h
        h_min = min(f2.size(2), f3.size(2), f4.size(2))
        ds2 = f2.size(2) - h_min
        ds3 = f3.size(2) - h_min
        ds4 = f4.size(2) - h_min
        # print(ds3//2, x_f2.size(2)-(ds3-ds3//2))
        # print(ds4//2, x_f2.size(2)-(ds4-ds4//2))

        x_f2 = f2[:, :, ds2//2:f2.size(2)-(ds2-ds2//2), :]
        x_f3 = f3[:, :, ds3//2:f3.size(2)-(ds3-ds3//2), :]
        x_f4 = f4[:, :, ds4//2:f4.size(2)-(ds4-ds4//2), :]
        # print(x_f2.size(), x_f3.size(), x_f4.size())

        # y = torch.cat((x_f2, x_f3, x_f4), dim=1)

        # multi-stage asymmetric addition
        c3 = x_f3.size(1)
        c4 = x_f4.size(1)

        # y = x_f2
        # y[:, 0:c3, ] += x_f3
        # y[:, 0:c4, ] += x_f4
        #
        y1 = x_f2[:, 0:c4, ] + x_f3[:, 0:c4, ] + x_f4
        y2 = x_f2[:, c4:c3, ] + x_f3[:, c4:c3, ]
        y3 = x_f2[:, c3:, ]
        y = torch.cat((y1, y2, y3), dim=1)
        # print(y.size())

        # y = self.conv_cat(y)
        # y = self.aspp(y)
        y = self.decoder_conv1(y)
        y = self.decoder_conv2(y)
        return y



if __name__ == '__main__':
    import numpy as np


