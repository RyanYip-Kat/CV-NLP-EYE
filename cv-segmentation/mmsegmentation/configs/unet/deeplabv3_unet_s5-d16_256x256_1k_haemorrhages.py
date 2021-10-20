_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/haemorrhages.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
optimizer = dict(type='SGD', lr=0.00005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=100)
checkpoint_config = dict(by_epoch=False, interval=20)

model = dict(test_cfg=dict(crop_size=(256, 256), stride=(170, 170)),
             auxiliary_head=dict(loss_decode=dict(type='FocalLoss', use_sigmoid=True, loss_weight=0.4)),
             decode_head=dict(loss_decode=dict(type='FocalLoss', use_sigmoid=True, loss_weight=0.4)))
evaluation = dict(metric='mDice')
