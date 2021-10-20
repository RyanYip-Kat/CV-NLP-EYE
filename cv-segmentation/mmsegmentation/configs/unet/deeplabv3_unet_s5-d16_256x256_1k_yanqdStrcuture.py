_base_ = [
     '../_base_/datasets/structure.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py','./deeplabv3_unet_s5-d16_mutiple_classes.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=2000)

model = dict(test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
evaluation = dict(interval=100, metric='mDice')
