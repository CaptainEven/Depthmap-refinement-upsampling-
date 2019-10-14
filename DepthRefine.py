# coding:utf-8

import copy
import os
import numpy as np
import cv2
import math
from numba import jit


def refine(num_iter=100, eta=0.5, sigma_c=10, sigma_s=10, L=10000, w_size=9):
    """
    @param sigma_c: sigma_color
    @param sigma_s: sigma_space
    implemention of "Spatial-Depth Super Resolution for Range Images"
    """
    # 读取原始图像(经过畸变矫正)和深度图
    src_path = './src.jpg'
    depthMap_path = './depth.jpg'  # Fusion的结果

    joint = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)  # 以原图为联合引导
    # depthMap = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)

    # if is resume
    depthMap = cv2.imread('./us_iter_8.jpg', cv2.IMREAD_GRAYSCALE)

    if joint is None or depthMap is None:
        print('[Err]: empty input image.')
        return

    DEPTH_H, DEPTH_W = depthMap.shape

    assert joint.shape == depthMap.shape

    # 都使用float32计算
    joint = joint.astype('float32')
    depthMap = depthMap.astype('float32')

    # 构造float32的cost volume和cost_cw(new cost volume): H, W, N
    cost_volume = np.zeros((DEPTH_H, DEPTH_W, 256), dtype='float32')
    cost_cw = np.zeros((DEPTH_H, DEPTH_W, 256), dtype='float32')

    THRESH = eta * L
    for iter_i in range(num_iter):
        for d in range(256):  # depth range
            tmp = np.empty((DEPTH_H, DEPTH_W))
            tmp.fill(d)

            cost_tmp = (tmp - depthMap) ** 2
            cost_tmp = np.where(cost_tmp < THRESH, cost_tmp, THRESH)
            cost_volume[:, :, d] = cost_tmp

            # 联合双边滤波
            cost_cw[:, :, d] = cv2.ximgproc.jointBilateralFilter(joint,
                                                                 cost_volume[:, :, d],
                                                                 -1,
                                                                 sigma_c, sigma_s)
            print('Depth hypothesis %d cost filtered' % d)

        # ------------------- 更新depth
        # get min cost along channels(depth hypotheses)
        min_cost = np.min(cost_cw, axis=2)  # f(x): min cost
        min_cost_depths = np.argmin(cost_cw, axis=2)  # x: min cost indices
        # print(min_cost_depths)

        # 亚像素深度估计 TODO: using numpy to accelerate python for loop
        for y in range(DEPTH_H):
            for x in range(DEPTH_W):
                f_d = cost_cw[y][x][min_cost_depths[y][x]]
                f_d_plus = cost_cw[y][x][min_cost_depths[y][x] + 1]
                f_d_minus = cost_cw[y][x][min_cost_depths[y][x] - 1]

                depth = min_cost_depths[y][x] - ((f_d_plus - f_d_minus) / (
                    2.0 * (f_d_plus + f_d_minus - 2.0 * f_d)))
                depthMap[y][x] = depth

        mat2show = copy.deepcopy(depthMap)
        
        mat2show = mat2show.astype('uint8')
        cv2.imwrite('./us_iter_%d.jpg' % (iter_i + 1 + 8), mat2show)
        print('=> Iter %d done\n' % (iter_i + 1))


if __name__ == '__main__':
    refine()
    print('=> Test done.')
