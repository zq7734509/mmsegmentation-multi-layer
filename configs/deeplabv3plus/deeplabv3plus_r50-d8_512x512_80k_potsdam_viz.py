_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/potsdam_600_viz.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoderViz',
    decode_head=dict(num_classes=5), auxiliary_head=dict(num_classes=5)
    )
test_cfg = dict(mode='whole')
optimizer = dict(type='SGD', lr=1.1e-4, momentum=0.9, weight_decay=0.0005)