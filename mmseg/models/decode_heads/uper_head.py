import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch.nn.functional as F
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .aspp_head import ASPPHead, ASPPModule

class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


class DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, dilations, **kwargs):
        temp_kwargs = kwargs
        temp_in_index = temp_kwargs["in_index"][-1]
        temp_in_channels = temp_kwargs["in_channels"][-1]
        temp_kwargs["in_index"] = temp_in_index
        temp_kwargs["in_channels"] = temp_in_channels

        super(DepthwiseSeparableASPPHead, self).__init__(dilations, **temp_kwargs)
        assert c1_in_channels >= 0
        self.aspp_depthwise_separable_modules = DepthwiseSeparableASPPModule(
            dilations=dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.x_bottleneck = ConvModule(
                self.in_channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_depthwise_separable_modules(x))

        del aspp_outs[1]
        x = self.x_bottleneck(x)
        aspp_outs.append(x)
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)

        return output

@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), dilations=(1, 2, 3, 4), c1_in_channels=96, c1_channels=48, **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.use_SE_net = False
        self.use_ASPP = False
        self.use_boundary = False
        if self.use_ASPP:
            # ASPP Module
            self.aspp_modules = DepthwiseSeparableASPPHead(c1_in_channels,
                                                           c1_channels,
                                                           dilations,
                                                           **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        # self.psp_bottleneck = ConvModule(
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.use_SE_net:
            self.fc1 = ConvModule(
                len(self.in_channels) * self.channels,
                len(self.in_channels) // 2 * self.channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

            self.fc2 = ConvModule(
                len(self.in_channels) // 2 * self.channels,
                len(self.in_channels) * self.channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None)

            self.DWS = DepthwiseSeparableConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        if self.use_boundary:
            self.fc1 = ConvModule(
                self.in_channels[0] + 2 * self.channels,
                self.in_channels[0] // 2 + self.channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

            self.fc2 = ConvModule(
                self.in_channels[0] // 2 + self.channels,
                self.in_channels[0] + 2 * self.channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None)

            self.boudary_bottleneck_1x1 = ConvModule(
                self.in_channels[0] + 2 * self.channels,
                self.channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.boudary_bottleneck_3x3 = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        # output = self.psp_bottleneck(psp_outs)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.use_ASPP:
            laterals.append(self.aspp_modules(inputs))
        else:
            laterals.append(self.psp_forward(inputs))


        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs_concat = torch.cat(fpn_outs, dim=1)

        if self.use_SE_net:
            fpn_weight = F.adaptive_avg_pool2d(fpn_outs_concat, 1)
            fpn_weight = self.fc1(fpn_weight)
            fpn_weight = self.fc2(fpn_weight)
            fpn_weight = torch.sigmoid(fpn_weight)
            fpn_outs_concat = fpn_outs_concat * fpn_weight
            fpn_outs_concat = fpn_outs_concat.contiguous()

        output = self.fpn_bottleneck(fpn_outs_concat)
        # output = self.DWS(output)

        if self.use_boundary:
            boundary_concat = []
            boundary_concat.append(inputs[0])       # 原始的stage1, channel数为128
            boundary_concat.append(fpn_outs[1])     # 经过逐像素相加, resize, 通道数不变的3x3卷积后的第二个stage, channel数为512
            boundary_concat.append(fpn_outs[3])     # 经过resize, 通道数不变的3x3卷积后的第二个stage, channel数为512
            boundary_concat = torch.cat(boundary_concat, dim=1)     # 在通道维度上进行堆叠

            # 进行SE操作
            boundary_channel_weight = F.adaptive_avg_pool2d(boundary_concat, 1)
            boundary_channel_weight = self.fc1(boundary_channel_weight)
            boundary_channel_weight = self.fc2(boundary_channel_weight)
            boundary_channel_weight = torch.sigmoid(boundary_channel_weight)
            boundary_concat = boundary_concat * boundary_channel_weight
            boundary_concat = boundary_concat.contiguous()

            # 进行降维和稳定操作
            boundary_concat = self.boudary_bottleneck_1x1(boundary_concat)
            boundary_concat = self.boudary_bottleneck_3x3(boundary_concat)
            output_binary = self.cls_binary_seg(boundary_concat)
        else:
            output_binary = None
        output_seg = self.cls_seg(output)

        return output_seg, output_binary
