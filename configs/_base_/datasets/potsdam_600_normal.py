# dataset settings
dataset_type = 'POTSDAMDataset'
data_root = 'data/mydataset_Potsdam_600_normal'

# add input
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, grey_mean=[116.28], grey_std=[57.375], grey_to_rgb=False)
crop_size = (300, 300)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),

    # add input
    # dict(type='LoadIR'),
    # dict(type='LoadDSM'),
    dict(type='Resize', img_scale=(600, 600), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),

    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),

    # add input
    # dict(type='LoadIR'),
    # dict(type='LoadDSM'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 600),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        # add input
        # ir_dir='ndvi_dir/train',
        # dsm_dir='ndsm_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        # # add input
        # ir_dir='ndvi_dir/val',
        # dsm_dir='ndsm_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        # add input
        # ir_dir='ndvi_dir/val',
        # dsm_dir='ndsm_dir/val',
        pipeline=test_pipeline))
