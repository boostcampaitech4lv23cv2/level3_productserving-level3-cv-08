_base_ = [
    './models/segformer_mit-b0.py',
    './datasets/dataset.py',
    './schedules/schedule.py',
    './runtime.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=5))

data = dict(
    val=dict(
        data_root='/opt/ml/input/data/car_damage_data',
        img_dir='mmseg/img_dir/val_small',
        ann_dir='mmseg/ann_dir/val_small'))