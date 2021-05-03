import cv2
import math
import numpy as np

from network import im_transform
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier

NUM_KEYPOINTS = 18
NUM_LIMBS = 19


def _get_outputs(multiplier, img, heatmaps, pafs, numkeypoints=NUM_KEYPOINTS, numlims=NUM_LIMBS):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], numkeypoints+1))
    paf_avg = np.zeros((img.shape[0], img.shape[1], numlims*2))

    heatmaps = heatmaps.transpose(0, 2, 3, 1)
    pafs = pafs.transpose(0, 2, 3, 1)
    # print('heatmaps', heatmaps.shape)

    # print('len(multiplier)', len(multiplier))
    for m in range(len(multiplier)):

        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = im_transform.crop_with_factor_kik(img, inp_size, factor=8, is_ceil=True)
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


def viz_keypoints(oriImg, p_heatmap, p_paf, name):
    # multiplier = get_multiplier(oriImg, 368.)

    # scale_search = [1.]
    # multiplier = [x * 481. / float(oriImg.shape[0]) for x in scale_search]
    multiplier = [1.]

    paf, heatmap = _get_outputs(multiplier, oriImg, p_heatmap, p_paf, numkeypoints=NUM_KEYPOINTS, numlims=NUM_LIMBS)

    # Given a (grayscale) image, find local maxima whose value is above a given threshold (param['thre1'])
    # Criterion 1: At least 80% of the intermediate points have a score higher than thre2
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    to_plot, canvas, joint_list, person_to_joint_assoc = decode_pose(oriImg, param, heatmap, paf)

    cv2.imwrite(name, canvas)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def viz_featuremaps(featuremaps, channelfirst=True, title='viz'):
    print('featuremaps shape:',featuremaps.shape)
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