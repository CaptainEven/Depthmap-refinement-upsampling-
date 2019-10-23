# coding:utf-8

import copy
import os
import numpy as np
import cv2
import math


def Test(num_iter=10, eta=0.5, sigma_c=10.0, sigma_s=10.0, L=256):
    """
    @param sigma_c: sigma_color
    @param sigma_s: sigma_space
    implemention of "Spatial-Depth Super Resolution for Range Images"
    """
    # 读取原始图像(经过畸变矫正)or结构增强的原图和深度图
    src_path = './src.jpg'
    depthMap_path = './depth2.jpg'  # Fusion的结果

    # 以原图为联合引导or结构增强的原图
    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)  
    depthMap = cv2.imread(depthMap_path, cv2.IMREAD_GRAYSCALE)

    # is_resume true
    # depthMap = cv2.imread('./iter_2_10.jpg', cv2.IMREAD_GRAYSCALE)

    if src is None or depthMap is None:
        print('[Err]: empty input image.')
        return

    DEPTH_H, DEPTH_W = depthMap.shape

    assert src.shape == depthMap.shape

    # ------------使用原图作为引导
    joint = src
    # ------------

    # ------------计算sobel, 用二阶sobel代替原图
    # sobel_x = cv2.Sobel(src, cv2.CV_16S, 1, 0, cv2.BORDER_DEFAULT)
    # sobel_y = cv2.Sobel(src, cv2.CV_16S, 0, 1, cv2.BORDER_DEFAULT)

    # sobel_xx = cv2.Sobel(sobel_x, cv2.CV_16S, 1, 0, cv2.BORDER_DEFAULT)
    # sobel_yy = cv2.Sobel(sobel_y, cv2.CV_16S, 0, 1, cv2.BORDER_DEFAULT)

    # sobel_xx = cv2.convertScaleAbs(sobel_xx)
    # sobel_yy = cv2.convertScaleAbs(sobel_yy)

    # joint = cv2.addWeighted(sobel_xx, 0.5, sobel_yy, 0.5, 0.0)
    # ------------

    # ---------Laplace of Gaussian算子平滑纹理丰富的区域并计算二阶梯度
    # blur = cv2.GaussianBlur(src, (3, 3), 0)
    # joint = cv2.Laplacian(blur, cv2.CV_16S, (3, 3))
    # joint = cv2.convertScaleAbs(joint)  # 如果不为了显示,这个应不应该计算?

    # joint_ = cv2.resize(joint,
    #                     (int(joint.shape[1]*0.3), int(joint.shape[0]*0.3)),
    #                     cv2.INTER_CUBIC)
    # cv2.imshow('log of src', joint_)
    # cv2.waitKey()
    # ------------

    # 图像数据都转换为float32计算
    joint = joint.astype('float32')
    depthMap = depthMap.astype('float32')

    # --------------------------
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

            # 调用opencv联合双边滤波
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

        # 亚像素深度估计
        for y in range(DEPTH_H):
            for x in range(DEPTH_W):
                int_depth = min_cost_depths[y][x]

                if int_depth + 1 > 255 or int_depth - 1 < 0:
                    depthMap[y][x] = float(int_depth)

                else:
                    f_d = cost_cw[y][x][int_depth]
                    f_d_plus = cost_cw[y][x][int_depth + 1]
                    f_d_minus = cost_cw[y][x][int_depth - 1]
    
                    sub_depth = int_depth - ((f_d_plus - f_d_minus) / (
                        2.0 * (f_d_plus + f_d_minus - 2.0 * f_d)))
                    depthMap[y][x] = sub_depth

        # 每个iteration对结果进一步过滤？

        mat2show = copy.deepcopy(depthMap)
        mat2show = np.round(mat2show)
        mat2show = mat2show.astype('uint8')
        cv2.imwrite('./iter_depth2_%d.jpg' % (iter_i + 1), mat2show)
        print('=> iter %d for depth2 done\n' % (iter_i + 1))


if __name__ == '__main__':
    Test()
    print('=> Test done.')
