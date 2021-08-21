# Copyright (c) SenseTime. All Rights Reserved.

import numpy as np
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """

    :param corner: two points represent the lower left corner and the upper right corner respectively
    :return:center: center point coordinate, width,height
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, w, y, h


def center2corner(center):
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        x2 = x + w * 0.5
        y1 = y - h * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def IoU(rect1, rect2):
    # IoU = area of overlap / area of union
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(x1, tx1)
    yy1 = np.maximum(y1, ty1)
    xx2 = np.minimum(x2, tx2)
    yy2 = np.minimum(y2, ty2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area_overlap = ww * hh
    area_union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - area_overlap

    return area_overlap / area_union


def cxy_wh_2_rect(pos, sz):
    """
    :param pos: center point
    :param sz: [w, h]
    :return: x1,y1,w,h
    """
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])


def get_axis_aligned_bbox(region):
    """
    :param region:
    :return: center point,w,h
    """
    nv = region.size
    # 如果是四角坐标，即可能不平行于图像
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        # 计算外接矩阵的左下角和右上角坐标
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        # 平行四边形面积
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        # 外接矩形面积
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0],
        y = region[1]
        w = region[2]
        h = region[3]

        cx = x + w / 2
        cy = y + h / 2

    return cx, cy, w, h
