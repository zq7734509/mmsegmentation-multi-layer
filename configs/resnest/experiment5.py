_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_potsdam_boundary_loss.py'
model = dict(
    pretrained='open-mmlab://resnest50',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True))
