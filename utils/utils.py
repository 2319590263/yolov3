import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from torchvision.ops import nms


def predict_transform(prediction, inp_dim, anchors, num_classes):
    """
    :param prediction:      预测结果
    :param inp_dim:         图像大小
    :param anchors:         先验框
    :param num_classes:     类别总数

    假设:
        prediction.shape = 64,13,13,75
        inp_dim = 416
        anchors = [[116, 90], [156, 198], [373, 326]]
        num_classes = 20

    """
    CUDA = torch.cuda.is_available()
    batch_size = prediction.size(0)  # 每个batch的样本数量        64
    stride = inp_dim // prediction.size(2)  # 网格步长                  32
    grid_size = inp_dim // stride  # 网格大小                  13
    bbox_attrs = 5 + num_classes  # 先验框参数                25
    num_anchors = len(anchors)  # 先验框数量                3

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors,
                                 grid_size * grid_size)  # prediction.shape = 64,75,169
    prediction = prediction.transpose(1, 2).contiguous()  # prediction.shape = 64,169,75
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors,
                                 bbox_attrs)  # prediction.shape = 64,507,80
    """
            对先验框进行处理
            处理完后结果为: anchors = [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]
    """
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    """
        对 (x,y) 坐标和 objectness 分数执行 Sigmoid 函数操作。
    """
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    """
        将网格偏移添加到中心坐标预测中
    """
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)  # 生成俩种坐标矩阵

    x_offset = torch.FloatTensor(a).view(-1, 1)  # 调整为:[[0,1,2,...,0,1,2,...]]
    y_offset = torch.FloatTensor(b).view(-1, 1)  # 调整为:[[0,0,0,...,1,1,1,...]]

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # 将x_offset和y_offset按1维度并行拼接在一起然后按1维度重复类别总数次最终结果为 假设数据结果:shape = 1, 13520, 2
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    if CUDA:
        # x_y_offset = x_y_offset.cuda()
        prediction = prediction.cuda()

    prediction[:, :, :2] += x_y_offset  # 给prediction中的x,y加上偏移
    """
        将锚点应用到边界框维度中
    """
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    """
        将 sigmoid 激活函数应用到类别分数中
    """
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    """
        将检测图的大小调整到与输入图像大小一致
    """
    prediction[:, :, :4] = (prediction[:, :, :4]*stride)/inp_dim

    return prediction


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        # ----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        # ----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # ------------------------------------------#
        #   获得预测结果中包含的所有种类
        # ------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # ------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            # ------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            # ------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            # ------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]

            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

    return output

if __name__ == "__main__":
    out0 = torch.rand((1, 52, 52, 75))
    out1 = torch.rand((1, 26, 26, 75))
    out2 = torch.rand((1, 13, 13, 75))
    outs = [out0, out1, out2]
    predictions = []
    for out in outs:
        predictions.append(predict_transform(out, 416, [[116, 90], [156, 198], [373, 326]], 20, False))
