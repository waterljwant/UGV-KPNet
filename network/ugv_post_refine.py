#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch

# TH_SCORE = 0.5
# TH_SCORE_AVG = 0.5

# ---- Better
TH_SCORE = 0.42
TH_SCORE_AVG = 0.48

TH_DIST = 5

# ---------------------------------------------------------
# 将卡板信息进行后处理
#

def pallet_post_kp_list(joint_list_per_joint_type):
    # print(len(joint_list_per_joint_type), joint_list_per_joint_type)
    # jointlist = np.asarray(joint_list_per_joint_type)

    kn = np.zeros(4)
    for i in range(4):
        kn[i] = joint_list_per_joint_type[i].shape[0]
    # print(kn)
    n_avg = np.mean(kn)
    n_avg = round(n_avg)
    if n_avg <= 1:
        return joint_list_per_joint_type

    new_list = list()
    num_removed = 0
    for i in range(4):
        joint_type_i = joint_list_per_joint_type[i]
        joint_type_i[:, 3] = joint_type_i[:, 3] - num_removed
        if kn[i] > n_avg:  # TODO， 目前只能删除多余的一个点
            min_index = np.argmin(joint_type_i[:, 2])
            # if joint_type_i[min_index, ]  # TODO: score < T, del
            # print('11', joint_type_i)
            joint_type_i = np.delete(joint_type_i, min_index, 0)
            # print('22', joint_type_i)
            # print('min_index', min_index)
            # update ID
            num_removed += 1
            joint_type_i[min_index:, 3] = joint_type_i[min_index:, 3] - 1
            # print('33', joint_type_i)

        new_list.append(joint_type_i)

    return new_list


# 这种后处理，必须是在只检测完整卡板的情况下使用。目前边缘点也为卡板真实点，可以这样处理，后续将边缘点视为虚拟点，则不能这样处理
# 删除点的数量较多的类别中得分最低点
def pallet_post_kp_array(candidate_kp):
    # candidate_kp [x, y, score, ID, type]
    new_candidate_kp = np.copy(candidate_kp)
    # print('------------------------------------')
    # print(candidate_kp)
    # --- 删除得分较低的点
    # mscore = np.mean(candidate_kp[:, 2])

    kn = np.zeros(4)
    for i in range(4):
        kn[i] = np.sum(candidate_kp[:, 4] == i)

    # print('kn', kn)
    n_avg = np.mean(kn)
    n_avg = round(n_avg)
    # print(n_avg, round(n_avg))
    kp_id = 0
    for i in range(4):  # loop for each type
        idx = np.where(candidate_kp[:, 4] == i)  # type
        print(i, idx, kn[i], kp_id)
        # min_score = np.amin(candidate_kp[idx, 2])
        min_index = np.argmin(candidate_kp[idx, 2])
        # print('min_score', min_score, 'min_index', min_index)
        if kn[i] > n_avg:  # TODO， 目前只能删除多余的一个点
            kp_id_to_remove = kp_id + min_index
            print('kp_id_to_remove', kp_id_to_remove)
            new_candidate_kp = np.delete(new_candidate_kp, kp_id_to_remove, 0)
            kp_id = kp_id - 1
        # ----
        kp_id += kn[i]  # point to next type
    # Reset ID
    for i in range(new_candidate_kp.shape[0]):
        new_candidate_kp[i, 3] = i

    return new_candidate_kp


def pallet_post_3to4th(candidate_kp, limbs, pallets):
    pallets_num = len(pallets)

    # if flag:
    #     print(pallets_num, ' pallets ------------------------------ ')
    #     print(pallets)
    #     print(candidate_kp)

    # 通过3个点，补齐另一个点
    kp = np.ones((pallets_num, 8)) * -1.
    kp_s = np.zeros((pallets_num, 4))
    for p in range(pallets_num):
        cnt = 0
        ms = -1
        for i in range(4):
            idx = pallets[p, i]
            if idx >= 0:
                cnt += 1
                idx = idx.astype(np.int32)
                kp[p, i * 2] = candidate_kp[idx, 0]
                kp[p, i * 2 + 1] = candidate_kp[idx, 1]
                kp_s[p, i] = candidate_kp[idx, 2]
            else:
                ms = i
        if cnt == 3:
            # print('cnt == 3, missing key point type: {}'.format(ms))
            idxms = ms
            # a = 3 - idxms
            # b = idxms + 2 if idxms < 2 else idxms - 2
            # c = 1 - idxms if idxms < 2 else 5 - idxms

            if ms == 0:
                a, b, c = 3, 2, 1  # v0 = v3 - v2 + v1
            elif ms == 1:
                a, b, c = 2, 3, 0  # v1 = v2 -v3 + v0
            elif ms == 2:
                a, b, c = 1, 0, 3  # v2 = v1 - v0 + v3
            elif ms == 3:
                a, b, c = 0, 1, 2  # v3 = v0 - v1 + v2

            # print(kp_s[p, a], kp_s[p, b], kp_s[p, c])
            if kp_s[p, a] > TH_SCORE and kp_s[p, b] > TH_SCORE and kp_s[p, c] > TH_SCORE \
                    and kp_s[p, a] + kp_s[p, b] + kp_s[p, c] > 3. * TH_SCORE_AVG:
                # print('Good {}'.format(ms))
                new_kp_x = kp[p, a * 2]     - kp[p, b * 2]     + kp[p, c * 2]
                new_kp_y = kp[p, a * 2 + 1] - kp[p, b * 2 + 1] + kp[p, c * 2 + 1]
                kp[p, idxms * 2]     = new_kp_x
                kp[p, idxms * 2 + 1] = new_kp_y

                # 在candidate中，查找与该点接近的点, 修改该点的信息
                flag_find_candidate = False
                v_4th = np.array([new_kp_x, new_kp_y])
                for idx_k in range(len(candidate_kp)):
                    v_kp = np.array([candidate_kp[idx_k, 0], candidate_kp[idx_k, 1]])  # TODO, check type
                    # if flag:
                    #     print(np.linalg.norm(v_4th - v_kp))
                    if np.linalg.norm(v_4th - v_kp) < TH_DIST:
                        # score
                        # print('candidate: idx_k=', idx_k, 'score=', candidate_kp[idx_k, 2])
                        if candidate_kp[idx_k, 2] > TH_SCORE_AVG:
                            pallets[p, idxms] = idx_k  # 更新当前卡板中缺失的那个点
                            flag_find_candidate = True
                        else:
                            # print('score too low')
                            pass
                # 如果没找到怎么办？
                if not flag_find_candidate:
                    # print('Not Found')
                    # 自己补点 ?  扩大搜索范围
                    pass
                # 判断补的点是否超出图像
            # ---------------------------------------------------------------
            '''
            t = TH_SCORE
            if ms == 0:
                idxms = ms
                a, b, c = 3, 2, 1  # v0 = v3 - v2 + v1
                if kp_s[p, a] > t and kp_s[p, b] > t and kp_s[p, c] > t:
                    print('Goood  {}'.format(ms))
                    new_kp_x = kp[p, a * 2] - kp[p, b * 2] + kp[p, c * 2]
                    new_kp_y = kp[p, a * 2 + 1] - kp[p, b * 2 + 1] + kp[p, c * 2 + 1]
                    kp[p, idxms * 2] = new_kp_x
                    kp[p, idxms * 2 + 1] = new_kp_y

                    # 在candidate中，查找与该点接近的点, 修改该点的信息
                    flag_find_candidate = False
                    v_4th = np.array([new_kp_x, new_kp_y])
                    for idx_k in range(len(candidate_kp)):
                        v_kp = np.array([candidate_kp[idx_k, 0], candidate_kp[idx_k, 1] ])  # TODO, check type
                        if np.linalg.norm(v_4th - v_kp) < TH_DIST:
                            pallets[p, idxms] = idx_k  # 更新当前卡板中缺失的那个点
                            print('candidate: idx_k=', idx_k, 'score=', candidate_kp[idx_k, 2])
                            print(candidate_kp)
                            flag_find_candidate = True
                    # 如果没找到怎么办？
                    if not flag_find_candidate:
                        print('Not Found')
                    # 判断补的点是否超出图像
            if ms == 1:
                idxms = ms
                a, b, c = 2, 3, 0  # v1 = v2 -v3 + v0
                if kp_s[p, a] > t and kp_s[p, b] > t and kp_s[p, c] > t:
                    print('Goood  {}'.format(ms))
                    new_kp_x = kp[p, a * 2] - kp[p, b * 2] + kp[p, c * 2]
                    new_kp_y = kp[p, a * 2 + 1] - kp[p, b * 2 + 1] + kp[p, c * 2 + 1]
                    kp[p, idxms * 2] = new_kp_x
                    kp[p, idxms * 2 + 1] = new_kp_y
                    # 在candidate中，查找与该点接近的点, 修改该点的信息
                    v_4th = np.array([new_kp_x, new_kp_y])
                    for idx_k in range(len(candidate_kp)):
                        v_kp = np.array([candidate_kp[idx_k, 0], candidate_kp[idx_k, 1] ])  # TODO, check type
                        if np.linalg.norm(v_4th - v_kp) < TH_DIST:
                            pallets[p, idxms] = idx_k  # 更新当前卡板中缺失的那个点
                            # print('candidate: idx_k=', idx_k, 'score=', candidate_kp[idx_k, 2])
                            # print(candidate_kp)
                    # 判断补的点是否超出图像
            if ms == 2:
                idxms = ms
                a, b, c = 1, 0, 3  # v2 = v1 - v0 + v3
                if kp_s[p, a] > t and kp_s[p, b] > t and kp_s[p, c] > t:
                    print('Goood  {}'.format(ms))
                    new_kp_x = kp[p, a * 2] - kp[p, b * 2] + kp[p, c * 2]
                    new_kp_y = kp[p, a * 2 + 1] - kp[p, b * 2 + 1] + kp[p, c * 2 + 1]
                    kp[p, idxms * 2] = new_kp_x
                    kp[p, idxms * 2 + 1] = new_kp_y

                    # 在candidate中，查找与该点接近的点, 修改该点的信息
                    v_4th = np.array([new_kp_x, new_kp_y])
                    for idx_k in range(len(candidate_kp)):
                        v_kp = np.array([candidate_kp[idx_k, 0], candidate_kp[idx_k, 1] ])  # TODO, check type
                        if np.linalg.norm(v_4th - v_kp) < TH_DIST:
                            pallets[p, idxms] = idx_k  # 更新当前卡板中缺失的那个点
                            # print('candidate: idx_k=', idx_k, 'score=', candidate_kp[idx_k, 2])
                            # print(candidate_kp)
                    # 判断补的点是否超出图像
            if ms == 3:
                idxms = ms
                a, b, c = 0, 1, 2  # v3 = v0 - v1 + v2
                if kp_s[p, a] > t and kp_s[p, b] > t and kp_s[p, c] > t:
                    print('Goood  {}'.format(ms))
                    new_kp_x = kp[p, a * 2] - kp[p, b * 2] + kp[p, c * 2]
                    new_kp_y = kp[p, a * 2 + 1] - kp[p, b * 2 + 1] + kp[p, c * 2 + 1]
                    kp[p, idxms * 2] = new_kp_x
                    kp[p, idxms * 2 + 1] = new_kp_y

                    # 在candidate中，查找与该点接近的点, 修改该点的信息
                    v_4th = np.array([new_kp_x, new_kp_y])
                    for idx_k in range(len(candidate_kp)):
                        v_kp = np.array([candidate_kp[idx_k, 0], candidate_kp[idx_k, 1] ])  # TODO, check type
                        if np.linalg.norm(v_4th - v_kp) < TH_DIST:
                            pallets[p, idxms] = idx_k  # 更新当前卡板中缺失的那个点
                            # print('candidate: idx_k=', idx_k, 'score=', candidate_kp[idx_k, 2])
                            # print(candidate_kp)
                    # 判断补的点是否超出图像
            '''
    # print(pallets)
    return pallets
