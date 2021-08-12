_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_potsdam.py'
model = dict(
    pretrained='open-mmlab://resnest50',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPDnlHead2',
        dilations=(1, 6, 12, 18),
        reduction=2,
        use_scale=True,
        mode='embedded_gaussian')
)
