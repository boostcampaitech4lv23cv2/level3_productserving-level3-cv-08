_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/ade20k.py',
    './default_runtime.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(pretrained=checkpoint, decode_head=dict(num_classes=2))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=250000)
evaluation = dict(
    interval=500,
    metric='mIoU',
    save_best='mIoU',
    pre_eval=True,
    by_epoch=False)
gpu_ids = [0]
data = dict(samples_per_gpu=32, workers_per_gpu=2)
