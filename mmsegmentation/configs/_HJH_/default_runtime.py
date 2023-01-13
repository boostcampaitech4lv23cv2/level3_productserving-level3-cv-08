log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMSegWandbHook',
            init_kwargs=dict(project='fianl', name='segformer b0 -damage'),
            num_eval_images=0)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
auto_resume = False
