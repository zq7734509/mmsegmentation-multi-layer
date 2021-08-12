_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_boundary_loss.py', '../_base_/datasets/hengyang_600_normal.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
test_cfg = dict(mode='whole')