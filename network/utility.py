#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch


from PIL import Image, ImageDraw, ImageFont
def show_img(meta, img, filename, text='GT'):
    img = img[:, :, ::-1]
    img = Image.fromarray(img)

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


# input: heatmap_target:(bs, c, h, w)
def resample(heatmap_target, heat_mask, paf_target, paf_mask, neg_rate):
    np_ht = heatmap_target.numpy()
    np_paf = paf_target.numpy()
    # print(np_ht.shape, heat_mask.shape)
    bs, c_ht, h, w = np_ht.shape
    new_heat_mask = np.ones((bs, c_ht, h, w), dtype=np.float32)

    # ----- resample heat map
    for i in range(bs):
        # num_pos = np.sum(np.array(ht[i, 4, :, :] < 1.))  # neg: background == 1
        # num_neg = np.sum(np.array(ht[i, 4, :, :] == 1.))

        # num_ign = int(h * w * neg_rate)
        num_ign = int(np.sum(np.array(np_ht[i, 4, :, :] == 1.)) * neg_rate)

        # print('Resample HeatMap: neg_rate', neg_rate, 'num_ign', num_ign)

        bg = np_ht[i][4].reshape(-1)
        bg_neg_dix = np.where(bg == 1.)  #
        inds = np.random.choice(bg_neg_dix[0], num_ign, replace=False)

        cht_mask = np.array(heat_mask[i, 4, :, :]).reshape(-1)
        cht_mask[inds] = 0.  # ignore
        cht_mask = cht_mask.reshape(h, w)

        for idx_c in range(c_ht):
            # print('idx_c', idx_c)
            new_heat_mask[i, idx_c] = cht_mask
    # --
    heat_mask = torch.from_numpy(new_heat_mask)

    # ----- resample paf
    bs, c_paf, h, w = paf_target.shape

    num_ign = int(np.sum(np.array(np_paf == 0.)) * neg_rate)
    # print('Resample PAF: neg_rate', neg_rate, 'num_ign', num_ign)
    bg = np_paf.reshape(-1)
    bg_neg_dix = np.where(bg == 0.)  #
    inds = np.random.choice(bg_neg_dix[0], num_ign, replace=False)

    new_paf_mask = np.array(paf_mask).reshape(-1)
    new_paf_mask[inds] = 0.  # ignore
    new_paf_mask = new_paf_mask.reshape(bs, c_paf, h, w)
    # --
    paf_mask = torch.from_numpy(new_paf_mask)
    return heat_mask, paf_mask



import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set()


def viz_featuremaps(featuremaps, channelfirst=True, title='viz'):
    channels = featuremaps.shape[0] if channelfirst else featuremaps.shape[2]
    grids_num = math.ceil(math.sqrt(channels))

    for i in range(channels):
        ax = plt.subplot(grids_num, grids_num, i + 1)
        ax.set_title('#{}'.format(i))
        ax.axis('off')
        hm = sns.heatmap(featuremaps[i, :, :]) if channelfirst else sns.heatmap(featuremaps[:, :, i])
    # plt.title(title)
    plt.suptitle(title, fontsize=16)
    plt.show()