# coding=utf-8
import numpy as np


def py_cpu_nms(dets, thresh):
  """Pure Python NMS baseline."""
  # tl_x,tl_y,br_x,br_y及score
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  # 计算每个检测框的面积，并对目标检测得分进行降序排序
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []  # 保留框的结果集合
  while order.size > 0:
    i = order[0]
    keep.append(i)  # 保留该类剩余box中得分最高的一个
    # 计算最高得分矩形框与剩余矩形框的相交区域
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    # 计算相交的面积,不重叠时面积为0
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    # 保留IoU小于阈值的box; 如果阈值很大表示为同一个物体，阈值比较小则可能为其他物体
    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]  # 注意这里索引加了1,因为ovr数组的长度比order数组的长度少一个

  return keep


if __name__ == '__main__':
  dets = np.array([[100, 120, 170, 200, 0.98],
                   [20, 40, 80, 90, 0.99],
                   [20, 38, 82, 88, 0.96],
                   [200, 380, 282, 488, 0.9],
                   [19, 38, 75, 91, 0.8]])

  _keep = py_cpu_nms(dets, 0.5)
  print(_keep)
