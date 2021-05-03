"""
ugvkp Dataloader
jieli_cn@163.com
2019/1/11
"""
import os

try:
    import ujson as json
except ImportError:
    import json

from torchvision.transforms import ToTensor
from training.datasets.ugv_data.UGV_data_pipeline import UGVKeyPoints
from training.datasets.dataloader import sDataLoader


def get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride, preprocess,
               batch_size, params_transform, training=True, shuffle=True, num_workers=1, aug=False, classification = False):
    """ Build a COCO dataloader
    :param json_path: string, path to jso file
    :param datadir: string, path to coco data
    :returns : the data_loader
    """

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
        with open(json_path) as data_file:
            json_data = json.load(data_file)
        # data_this = json.load(data_file)
        # data = data_this['root']

    num_samples = len(json_data)
    train_indexes = []
    val_indexes = []
    for count in range(num_samples):
        if json_data[count]['isValidation'] != 0.:
            val_indexes.append(count)
        else:
            train_indexes.append(count)

    # print('train dataset len:', len(train_indexes), '  val dataset len:', len(val_indexes))

    # root = data_dir
    # root = os.path.dirname(json_path)

    kik_data = UGVKeyPoints(root=root,
                            index_list=train_indexes if training else val_indexes,
                            data=json_data, feat_stride=feat_stride,
                            preprocess=preprocess, transform=ToTensor(), params_transform=params_transform,
                            numkeypoints=4, numlims=4, aug=aug, classification=classification)  # Mod by Jie.

    data_loader = sDataLoader(kik_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader

