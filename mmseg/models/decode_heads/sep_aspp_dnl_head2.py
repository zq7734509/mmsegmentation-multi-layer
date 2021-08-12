import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
#from .aspp_head import ASPPHead, ASPPModule
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .dnl_head import DNLHead

class DNLModule(DNLHead):
    def __init__(self, reduction, use_scale, mode, dilations, **kwargs):
        super(DNLModule, self).__init__(reduction, use_scale, mode, **kwargs)
        self.in_channels = self.channels
        self.convs[0] = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=3 // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)


    def forward(self, inputs):
        x = inputs
        output = self.convs[0](x)
        output = self.dnl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        return output

@HEADS.register_module()
class DepthwiseSeparableASPPDnlHead2(DepthwiseSeparableASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution and DNL for Semantic Image
    Segmentation.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, reduction, use_scale, mode, **kwargs):
        super(DepthwiseSeparableASPPDnlHead2, self).__init__(c1_in_channels, c1_channels, **kwargs)
        assert c1_in_channels >= 0
        self.dnl_module = DNLModule(reduction, use_scale, mode, **kwargs)

        # 用于计算热力图
        self.return_dnl = None
        self.return_aspp = None
        self.return_seg = None

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        aspp_outs = [   # s1.对最后一层特征图进行全局平均池化, 然后缩放至与原特征图相同尺寸
            resize(
                self.image_pool(x),
                size=x.size()[2:],  # 获取特征图的宽高属性
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))      # s2.将特征分别通过四个深度可分离卷积模块进行处理, 并将处理结果接在上一步上采样的全局池化结果后
        aspp_outs = torch.cat(aspp_outs, dim=1)     # s3.将上述五个特征层在通道维度上进行堆叠
        output = self.bottleneck(aspp_outs)         # s4.将堆叠后的特征通过3*3的卷积操作进行降维，将特征的层数减少至堆叠后的5分之1,即堆叠前的层数
        if self.c1_bottleneck is not None:

            c1_output = self.c1_bottleneck(inputs[0])       # s5.将低层特征（resnet的四个stage中的第一个stage的结果）按照给定的参数进行降维操作
            output = resize(        # s6.将经过降维后的堆叠特征进行上采样, 得到与低层特征相同的宽高
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)  # s7.将上述宽高相同的特征进行通道维度的堆叠
        output = self.sep_bottleneck(output)    # s8.对上一步中的堆叠特征先后进行两个3*3的深度可分离卷积, 第一次是为了改变通道数至s4步骤中的结果通道数, 第二次的通道数保持不变

        self.return_aspp = output

        output = self.dnl_module(output)
        self.return_dnl = output

        output = self.cls_seg(output)   # s9.先进行dropout操作,再利用1*1的卷积操作将通道数由当前数量转换为预期类别总数,所得结果的宽高均为原图的1/4
        self.return_seg = output

        return output
