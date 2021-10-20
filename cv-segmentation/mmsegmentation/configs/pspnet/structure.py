_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
    '../_base_/datasets/structure.py'
]
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)


norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(num_classes=4, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=4, norm_cfg=norm_cfg),
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(depth=50, norm_cfg=norm_cfg)
)

runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=200, metric='mDice')

