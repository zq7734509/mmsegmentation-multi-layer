_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/potsdam_600_normal.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=5), auxiliary_head=dict(num_classes=5))
test_cfg = dict(mode='whole')
