# dataset settings
dataset_type = 'CustomDataset'
classes = (
    'Backgroud',
    'Scratched',
    'Crushed',
    'Breakage',
    'Separated'
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
#         flip=True,
#         flip_direction=['horizontal'],
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root='/opt/ml/input/data/car_damage_data',
        img_dir='data/training/src/damage',
        ann_dir='mmseg/ann_dir/train',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root='/opt/ml/input/data/car_damage_data',
        img_dir='data/validation/src/damage',
        ann_dir='mmseg/ann_dir/val',
        classes=classes,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root='/opt/ml/input/data/car_damage_data',
        img_dir='data/validation/src/damage',
        ann_dir='mmseg/ann_dir/val',
        classes=classes,
        pipeline=test_pipeline))
