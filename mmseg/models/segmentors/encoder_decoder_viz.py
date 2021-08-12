import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import cv2
import numpy as np
import mmseg.datasets.potsdam as dataset

@SEGMENTORS.register_module()
class EncoderDecoderViz(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoderViz, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)


        self.feature_map_count = 2
        self.origin_img_count = 0
        self.dnl_heatmap_index = 2
        self.aspp_heatmap_index = 2
        self.seg_heatmap_index = 2
        self.seg_output_count = 0

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoderViz, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if isinstance(out, tuple):
            out = out[0]
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def save_feature_map(self, features, origin_img):
        """the following part shows the feature map of each stage, using the 0, 1/4, 1/2, 3/4 and 1 channels."""
        origin_img_0 = origin_img[0, :, :, :].cpu().numpy()
        origin_img_1 = origin_img[1, :, :, :].cpu().numpy()
        # 1st stage:

        h,w,_ = origin_img_0.shape

        output_1 = np.zeros((4 * h, 6 * w, 3), np.uint8)
        output_2 = np.zeros((4 * h, 6 * w, 3), np.uint8)

        rate = 0.7

        for i in range(4):
            temp = features[i].data.cpu().numpy()
            n, c, _, _ = temp.shape

            max_value = temp[0, 0, :, :].max() * rate
            min_value = temp[0, 0, :, :].min()
            temp_map11 = (temp[0, 0, :, :] - min_value) / max_value

            max_value = temp[0, c//4, :, :].max() * rate
            min_value = temp[0, c//4, :, :].min()
            temp_map12 = (temp[0, c//4, :, :] - min_value) / max_value

            max_value = temp[0, c//2, :, :].max() * rate
            min_value = temp[0, c//2, :, :].min()
            temp_map13 = (temp[0, c//2, :, :] - min_value) / max_value

            max_value = temp[0, 3 * (c//4), :, :].max() * rate
            min_value = temp[0, 3 * (c//4), :, :].min()
            temp_map14 = (temp[0, 3 * (c//4), :, :] - min_value) / max_value

            max_value = temp[0, c-1, :, :].max() * rate
            min_value = temp[0, c-1, :, :].min()
            temp_map15 = (temp[0, c-1, :, :] - min_value) / max_value



            max_value = temp[1, 0, :, :].max() * rate
            min_value = temp[1, 0, :, :].min()
            temp_map21 = (temp[1, 0, :, :] - min_value) / max_value

            max_value = temp[1, c // 4, :, :].max() * rate
            min_value = temp[1, c // 4, :, :].min()
            temp_map22 = (temp[1, c // 4, :, :] - min_value) / max_value

            max_value = temp[1, c // 2, :, :].max() * rate
            min_value = temp[1, c // 2, :, :].min()
            temp_map23 = (temp[1, c // 2, :, :] - min_value) / max_value

            max_value = temp[1, 3 * (c // 4), :, :].max() * rate
            min_value = temp[1, 3 * (c // 4), :, :].min()
            temp_map24 = (temp[1, 3 * (c // 4), :, :] - min_value) / max_value

            max_value = temp[1, c - 1, :, :].max() * rate
            min_value = temp[1, c - 1, :, :].min()
            temp_map25 = (temp[1, c - 1, :, :] - min_value) / max_value

            heatmap_11 = np.uint8(255*temp_map11)
            heatmap_11 = cv2.resize(heatmap_11, (h, w))
            heatmap_11 = cv2.applyColorMap(heatmap_11, cv2.COLORMAP_HOT)
            heatmap_12 = np.uint8(255 * temp_map12)
            heatmap_12 = cv2.resize(heatmap_12, (h, w))
            heatmap_12 = cv2.applyColorMap(heatmap_12, cv2.COLORMAP_HOT)
            heatmap_13 = np.uint8(255 * temp_map13)
            heatmap_13 = cv2.resize(heatmap_13, (h, w))
            heatmap_13 = cv2.applyColorMap(heatmap_13, cv2.COLORMAP_HOT)
            heatmap_14 = np.uint8(255 * temp_map14)
            heatmap_14 = cv2.resize(heatmap_14, (h, w))
            heatmap_14 = cv2.applyColorMap(heatmap_14, cv2.COLORMAP_HOT)
            heatmap_15 = np.uint8(255 * temp_map15)
            heatmap_15 = cv2.resize(heatmap_15, (h, w))
            heatmap_15 = cv2.applyColorMap(heatmap_15, cv2.COLORMAP_HOT)

            heatmap_21 = np.uint8(255*temp_map21)
            heatmap_21 = cv2.resize(heatmap_21, (h, w))
            heatmap_21 = cv2.applyColorMap(heatmap_21, cv2.COLORMAP_HOT)
            heatmap_22 = np.uint8(255 * temp_map22)
            heatmap_22 = cv2.resize(heatmap_22, (h, w))
            heatmap_22 = cv2.applyColorMap(heatmap_22, cv2.COLORMAP_HOT)
            heatmap_23 = np.uint8(255 * temp_map23)
            heatmap_23 = cv2.resize(heatmap_23, (h, w))
            heatmap_23 = cv2.applyColorMap(heatmap_23, cv2.COLORMAP_HOT)
            heatmap_24 = np.uint8(255 * temp_map24)
            heatmap_24 = cv2.resize(heatmap_24, (h, w))
            heatmap_24 = cv2.applyColorMap(heatmap_24, cv2.COLORMAP_HOT)
            heatmap_25 = np.uint8(255 * temp_map25)
            heatmap_25 = cv2.resize(heatmap_25, (h, w))
            heatmap_25 = cv2.applyColorMap(heatmap_25, cv2.COLORMAP_HOT)

            output_1[i * h:(i+1) * h, 0:w] = origin_img_0;
            output_1[i * h:(i+1) * h, w:2 * w] = heatmap_11;
            output_1[i * h:(i+1) * h, 2 * w:3 * w] = heatmap_12;
            output_1[i * h:(i+1) * h, 3 * w:4 * w] = heatmap_13;
            output_1[i * h:(i+1) * h, 4 * w:5 * w] = heatmap_14;
            output_1[i * h:(i+1) * h, 5 * w:6 * w] = heatmap_15;

            output_2[i * h:(i+1) * h, 0:w] = origin_img_1;
            output_2[i * h:(i+1) * h, w:2 * w] = heatmap_21;
            output_2[i * h:(i+1) * h, 2 * w:3 * w] = heatmap_22;
            output_2[i * h:(i+1) * h, 3 * w:4 * w] = heatmap_23;
            output_2[i * h:(i+1) * h, 4 * w:5 * w] = heatmap_24;
            output_2[i * h:(i+1) * h, 5 * w:6 * w] = heatmap_25;

        index_str1 = str(self.feature_map_count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/feature_map/' + index_str1 + '.png'
        cv2.imwrite(output_path_str1, output_1)

        index_str2 = str(self.feature_map_count+1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/feature_map/' + index_str2 + '.png'
        cv2.imwrite(output_path_str2, output_2)
        self.feature_map_count += 2

    def save_dnl_origin_img(self, origin_img, label):
        origin_img_0 = origin_img[0, :, :, :].cpu().numpy()
        origin_img_1 = origin_img[1, :, :, :].cpu().numpy()

        index_str1 = str(self.origin_img_count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/dnl_origin_img/' + index_str1 + '.png'
        cv2.imwrite(output_path_str1, origin_img_0)

        index_str2 = str(self.origin_img_count + 1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/dnl_origin_img/' + index_str2 + '.png'
        cv2.imwrite(output_path_str2, origin_img_1)

        ####################################################################################
        label_0 = label[0, :, :, :].cpu().numpy()
        label_1 = label[1, :, :, :].cpu().numpy()

        label_0 = np.transpose(label_0, (1, 2, 0))
        label_0 = np.uint8(255 * (label_0 + 1) / 5)
        label_0 = cv2.applyColorMap(label_0, cv2.COLORMAP_JET)
        label_1 = np.transpose(label_1, (1, 2, 0))
        label_1 = np.uint8(255 * (label_1 + 1) / 5)
        label_1 = cv2.applyColorMap(label_1, cv2.COLORMAP_JET)

        index_str1 = str(self.origin_img_count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/dnl_origin_img/' + index_str1 + '_label.png'
        cv2.imwrite(output_path_str1, label_0)

        index_str2 = str(self.origin_img_count + 1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/dnl_origin_img/' + index_str2 + '_label.png'
        cv2.imwrite(output_path_str2, label_1)
        ####################################################################################
        self.origin_img_count += 2

    def save_boundary_origin_img(self, origin_img, label):
        ####################################################################################
        origin_img_0 = origin_img[0, :, :, :].cpu().numpy()
        origin_img_1 = origin_img[1, :, :, :].cpu().numpy()

        index_str1 = str(self.origin_img_count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/boundary_origin_img/' + index_str1 + '.png'
        cv2.imwrite(output_path_str1, origin_img_0)

        index_str2 = str(self.origin_img_count + 1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/boundary_origin_img/' + index_str2 + '.png'
        cv2.imwrite(output_path_str2, origin_img_1)
        ####################################################################################
        label_0 = label[0, :, :, :].cpu().numpy()
        label_1 = label[1, :, :, :].cpu().numpy()

        label_0 = np.transpose(label_0, (1, 2, 0))
        label_0 = np.uint8(255 * (label_0+1) / 5)
        label_0 = cv2.applyColorMap(label_0, cv2.COLORMAP_JET)
        label_1 = np.transpose(label_1, (1, 2, 0))
        label_1 = np.uint8(255 * (label_1 + 1) / 5)
        label_1 = cv2.applyColorMap(label_1, cv2.COLORMAP_JET)

        index_str1 = str(self.origin_img_count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/boundary_origin_img/' + index_str1 + '_label.png'
        cv2.imwrite(output_path_str1, label_0)

        index_str2 = str(self.origin_img_count + 1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/boundary_origin_img/' + index_str2 + '_label.png'
        cv2.imwrite(output_path_str2, label_1)
        ####################################################################################
        self.origin_img_count += 2


    def hook_dnl_fn(self, grad):
        # s1.将batch中的两张原图转移至cpu设备上, 并将tensor转换为numpy
        origin_img_0 = self.origin_img[0, :, :, :].cpu().numpy()
        origin_img_1 = self.origin_img[1, :, :, :].cpu().numpy()

        # s2.以channel为主维度, 计算各channel的梯度平均值
        mean_gradients = torch.mean(grad, dim=[0, 2, 3])  # mean_gradients是一个长度为channel的向量

        # s3.获取目标特征图的数据, &&需要注意的是:此处需要通过'.data'方法进行赋值, 否则会导致backward过程中的占位符错误问题&&
        feature = self.return_dnl.data

        # s4.特征图的每个channel乘以对应通道的梯度平均值
        for i in range(len(mean_gradients)):
            feature[:, i, :, :] *= mean_gradients[i]

        # s5.对处理后的特征图在channel维度上进行求平均, heatmap的维度为[batch,height,width]
        heatmap = torch.mean(feature, dim=1)

        # s6.将heatmap中的负值调整为0, 然后进行归一化, 方便后期拉伸至0~255
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # s7.分别对heatmap中batch维度所对应的两个子图进行处理:
        #    a.转移至cpu设备并转换为numpy
        #    b.缩放至与原图相同尺寸
        #    c.线性拉伸至0~255
        #    d.将灰度的heatmap转换为COLORMAP_JET色调的三通道彩色图
        heatmap_0 = heatmap[0, :, :].cpu().numpy()
        heatmap_0 = cv2.resize(heatmap_0, (origin_img_0.shape[1], origin_img_0.shape[0]))
        heatmap_0 = np.uint8(255 * heatmap_0)
        heatmap_0 = cv2.applyColorMap(heatmap_0, cv2.COLORMAP_JET)

        heatmap_1 = heatmap[1, :, :].cpu().numpy()
        heatmap_1 = cv2.resize(heatmap_1, (origin_img_1.shape[1], origin_img_1.shape[0]))
        heatmap_1 = np.uint8(255 * heatmap_1)
        heatmap_1 = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)

        # s9.构造两个个大图, 1行2列,存储4张子图, 每行代表batch中的一个子图的原图\热力图
        output_0 = np.zeros((origin_img_0.shape[0], 2*origin_img_0.shape[1], 3), np.uint8)
        output_0[:, 0:origin_img_0.shape[0]] = origin_img_0
        output_0[:, origin_img_0.shape[0]:] = heatmap_0

        output_1 = np.zeros((origin_img_0.shape[0], 2 * origin_img_0.shape[1], 3), np.uint8)
        output_1[:, 0:origin_img_0.shape[0]] = origin_img_1
        output_1[:, origin_img_0.shape[0]:] = heatmap_1

        # s10.将上面的图片按照索引序号存储至指定路径, 将索引数值转换为6位字符串, 位数不够则在左侧补0
        index_str_0 = str(self.dnl_heatmap_index).zfill(6)
        output_path_str_0 = '/home/allen/mmseg_viz/dnl_heatmap/' + index_str_0 + '.png'
        cv2.imwrite(output_path_str_0, output_0)
        index_str_1 = str(self.dnl_heatmap_index + 1).zfill(6)
        output_path_str_1 = '/home/allen/mmseg_viz/dnl_heatmap/' + index_str_1 + '.png'
        cv2.imwrite(output_path_str_1, output_1)

        self.dnl_heatmap_index += 2

    def hook_aspp_fn(self, grad):
        # s1.将batch中的两张原图转移至cpu设备上, 并将tensor转换为numpy
        origin_img_0 = self.origin_img[0, :, :, :].cpu().numpy()
        origin_img_1 = self.origin_img[1, :, :, :].cpu().numpy()

        # s2.以channel为主维度, 计算各channel的梯度平均值
        mean_gradients = torch.mean(grad, dim=[0, 2, 3])  # mean_gradients是一个长度为channel的向量

        # s3.获取目标特征图的数据, &&需要注意的是:此处需要通过'.data'方法进行赋值, 否则会导致backward过程中的占位符错误问题&&
        feature = self.return_aspp.data

        # s4.特征图的每个channel乘以对应通道的梯度平均值
        for i in range(len(mean_gradients)):
            feature[:, i, :, :] *= mean_gradients[i]

        # s5.对处理后的特征图在channel维度上进行求平均, heatmap的维度为[batch,height,width]
        heatmap = torch.mean(feature, dim=1)

        # s6.将heatmap中的负值调整为0, 然后进行归一化, 方便后期拉伸至0~255
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # s7.分别对heatmap中batch维度所对应的两个子图进行处理:
        #    a.转移至cpu设备并转换为numpy
        #    b.缩放至与原图相同尺寸
        #    c.线性拉伸至0~255
        #    d.将灰度的heatmap转换为COLORMAP_JET色调的三通道彩色图
        heatmap_0 = heatmap[0, :, :].cpu().numpy()
        heatmap_0 = cv2.resize(heatmap_0, (origin_img_0.shape[1], origin_img_0.shape[0]))
        heatmap_0 = np.uint8(255 * heatmap_0)
        heatmap_0 = cv2.applyColorMap(heatmap_0, cv2.COLORMAP_JET)

        heatmap_1 = heatmap[1, :, :].cpu().numpy()
        heatmap_1 = cv2.resize(heatmap_1, (origin_img_1.shape[1], origin_img_1.shape[0]))
        heatmap_1 = np.uint8(255 * heatmap_1)
        heatmap_1 = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)

        # s9.构造2个大图, 1行2列存储4张子图, 每行代表batch中的一个子图的原图\热力图\
        output_0 = np.zeros((origin_img_0.shape[0], 2 * origin_img_0.shape[1], 3), np.uint8)
        output_0[:, 0:origin_img_0.shape[0]] = origin_img_0
        output_0[:, origin_img_0.shape[0]:] = heatmap_0

        output_1 = np.zeros((origin_img_0.shape[0], 2 * origin_img_0.shape[1], 3), np.uint8)
        output_1[:, 0:origin_img_0.shape[0]] = origin_img_1
        output_1[:, origin_img_0.shape[0]:] = heatmap_1

        # s10.将上面的图片按照索引序号存储至指定路径, 将索引数值转换为6位字符串, 位数不够则在左侧补0
        index_str_0 = str(self.aspp_heatmap_index).zfill(6)
        output_path_str_0 = '/home/allen/mmseg_viz/aspp_heatmap/' + index_str_0 + '.png'
        cv2.imwrite(output_path_str_0, output_0)
        index_str_1 = str(self.aspp_heatmap_index + 1).zfill(6)
        output_path_str_1 = '/home/allen/mmseg_viz/aspp_heatmap/' + index_str_1 + '.png'
        cv2.imwrite(output_path_str_1, output_1)

        self.aspp_heatmap_index += 2

    def hook_seg_fn(self, grad):
        # s2.以channel为主维度, 计算各channel的梯度平均值
        # mean_gradients = torch.mean(grad, dim=[2, 3])  # mean_gradients是一个长度为channel的向量
        # index_str_0 = str(self.seg_heatmap_index).zfill(6)
        # index_str_1 = str(self.seg_heatmap_index+1).zfill(6)
        # output_path_str_0 = '/home/allen/mmseg_viz/seg_grad/' + index_str_0 + '_grad.txt'
        # output_path_str_1 = '/home/allen/mmseg_viz/seg_grad/' + index_str_1 + '_grad.txt'
        # mean_gradients_0 = mean_gradients.data.cpu().numpy()
        # np.savetxt(output_path_str_0, mean_gradients[0, :].data.cpu().numpy())
        # np.savetxt(output_path_str_1, mean_gradients[1, :].data.cpu().numpy())
        # self.seg_heatmap_index += 2

        mean_gradients = torch.mean(grad, dim=[0, 2, 3])  # mean_gradients是一个长度为channel的向量
        index_str = str(self.seg_heatmap_index).zfill(6)
        output_path_str = '/home/allen/mmseg_viz/seg_grad/' + index_str + '_grad.txt'
        mean_gradients_0 = mean_gradients.data.cpu().numpy()
        np.savetxt(output_path_str, mean_gradients.data.cpu().numpy())
        self.seg_heatmap_index += 2

    def save_seg_results(self,
                         result,
                         palette=None,
                         out_file_folder=None):
        n = len(result)
        for i in range(n):
            out_file = out_file_folder + str(self.seg_output_count).zfill(6) + '.png'
            seg = result[i]

            if palette is None:
                if self.PALETTE is None:
                    palette = np.random.randint(
                        0, 255, size=(len(self.CLASSES), 3))
                else:
                    palette = self.PALETTE
            palette = np.array(palette)

            assert palette.shape[1] == 3
            assert len(palette.shape) == 2
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            # convert to BGR
            color_seg = color_seg[..., ::-1]

            # img = img * 0.5 + color_seg * 0.5
            img = color_seg
            img = img.astype(np.uint8)
            if out_file is not None:
                cv2.imwrite(out_file, img)
            self.seg_output_count += 1

    def forward_train(self, img, img_metas, gt_semantic_seg, origin_img):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        ## "save_feature_map" shows the feature map of each stage, using the 0, 1/4, 1/2, 3/4 and 1 channels
        # self.save_feature_map(x,origin_img)

        ## "save_dnl_origin_img" saves the origin image for dnl_viz
        # self.save_dnl_origin_img(origin_img, gt_semantic_seg)

        ## "save_boundary_origin_img" saves the origin image for boundary_viz
        self.save_boundary_origin_img(origin_img, gt_semantic_seg)

        losses = dict()

        self.origin_img = origin_img.data

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)


        # self.return_aspp = self.decode_head.return_aspp
        # self.return_aspp.register_hook(self.hook_aspp_fn)
        # self.return_dnl = self.decode_head.return_dnl
        # self.return_dnl.register_hook(self.hook_dnl_fn)
        # self.return_seg = self.decode_head.return_seg
        # self.return_seg.register_hook(self.hook_seg_fn)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        pred_seg = self.simple_test(img, img_metas, rescale=False)
        self.save_seg_results(pred_seg, palette=dataset.POTSDAMDataset.PALETTE,
                              out_file_folder='/home/allen/mmseg_viz/seg_image/')

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        assert h_crop <= h_img and w_crop <= w_img, (
            'crop size should not greater than image size')
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        # flip = img_meta[0]['flip']
        # if flip:
        #     flip_direction = img_meta[0]['flip_direction']
        #     assert flip_direction in ['horizontal', 'vertical']
        #     if flip_direction == 'horizontal':
        #         output = output.flip(dims=(3, ))
        #     elif flip_direction == 'vertical':
        #         output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
