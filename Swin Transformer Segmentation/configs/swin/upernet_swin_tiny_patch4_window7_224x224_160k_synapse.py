_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/synapse.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# Model configuration for Synapse dataset
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
        num_classes=9,  # Background + 8 organs
        # Use Dice+CE combination loss for multi-class segmentation
        # Similar to ISIC2017 but adapted for 9 classes
        loss_decode=dict(
            type='DiceCELoss',
            dice_weight=1.0,      # Dice loss weight (handles class imbalance)
            ce_weight=1.0,         # CrossEntropy loss weight (provides gradient stability)
            smooth=1,              # Dice smoothing factor
            # Class weights: background=0.1, organs=0.9 (similar to ISIC2017)
            # For 9 classes: [background, aorta, gallbladder, spleen, left_kidney,
            #                right_kidney, liver, pancreas, stomach]
            class_weight=[0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            loss_weight=1.0,
            ignore_index=255      # Ignore boundary pixels
        )
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=9,
        # Auxiliary head also uses Dice+CE combination loss
        loss_decode=dict(
            type='DiceCELoss',
            dice_weight=1.0,
            ce_weight=1.0,
            smooth=1,
            class_weight=[0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            loss_weight=0.4,      # Auxiliary head weight
            ignore_index=255
        )
    ))

# Optimizer configuration
# Similar to ISIC2017 but may need adjustment for multi-class task
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

# Learning rate schedule
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=0.9,
                 min_lr=1e-6,
                 by_epoch=False)

# Data configuration
# Input size is 224x224 for Synapse (as in VM-UNet)
data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=2)

# Runner configuration
# 160000 iterations for full training
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)  # Save checkpoint every 16000 iterations
# Evaluation configuration with 3D volume support
# use_3d_eval=True: Use full 3D volume evaluation (accurate but slower, ~30-40 min per evaluation)
# use_3d_eval=False: Use 2D slice evaluation (fast but inaccurate, only first slice)
evaluation = dict(
    interval=16000,         # Evaluate every 16000 iterations
    metric='mDice',         # Primary metric
    use_3d_eval=True        # Enable 3D volume evaluation during training (accurate)
)

