_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/isic2017.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        # Use Dice+CE combined loss with class weights to handle class imbalance
        # Reference training result analysis: best performance at iter 80000 (mIoU 87.06%)
        loss_decode=dict(
            type='DiceCELoss',
            dice_weight=1.0,      # Dice loss weight (handles class imbalance)
            ce_weight=1.0,         # CrossEntropy loss weight (provides gradient stability)
            smooth=1,              # Dice smoothing factor
            class_weight=[0.1, 0.9],  # CE loss class weights: background=0.1, lesion=0.9
            loss_weight=1.0,
            ignore_index=255      # Ignore boundary pixels
        )
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2,
        # Auxiliary head also uses Dice+CE combined loss
        loss_decode=dict(
            type='DiceCELoss',
            dice_weight=1.0,
            ce_weight=1.0,
            smooth=1,
            class_weight=[0.1, 0.9],  # Also use class weights
            loss_weight=0.4,      # Auxiliary head weight
            ignore_index=255
        )
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
# Resume training from iter 80000, use smaller learning rate (1/3 to 1/2 of original)
# Reasons: 1) Already trained 50%, 2) New loss function requires more stable learning rate
optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

# Learning rate policy: use gentler decay
# Resume training from iter 80000, remaining 80000 iterations
# Use cosine annealing or smaller power value to make learning rate decay smoother
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=500,      # Reduce warmup (because resuming from middle)
                 warmup_ratio=1e-6,
                 power=0.8,            # Further reduce decay strength (originally 0.9)
                 min_lr=1e-6,          # Set minimum learning rate to avoid too small LR (originally 0.0)
                 by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# For 4 GPUs training, use SyncBN (default in base config)
data=dict(samples_per_gpu=2)

