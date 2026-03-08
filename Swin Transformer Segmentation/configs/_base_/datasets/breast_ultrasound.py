# dataset settings
dataset_type = 'BreastUltrasoundDataset'  # Use custom dataset class to handle special naming rules
data_root = '/home/lvyp/Breast_BUS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (224, 224)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Note: Breast Ultrasound labels are already in 0/1 format, no ConvertLabels needed
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
# Validation pipeline: labels are obtained via get_gt_seg_maps during evaluation, no LoadAnnotations needed in pipeline
# Note: validation set is forced to test_mode=True in train.py, so pipeline cannot contain LoadAnnotations
# Use MultiScaleFlipAug to generate required meta info (e.g., flip field), consistent with test_pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,  # No flipping during validation
        transforms=[
            dict(type='Resize', keep_ratio=True),  # Keep aspect ratio
            dict(type='RandomFlip'),  # Although flip=False, RandomFlip will set flip field
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Test pipeline: no need to load labels (for inference)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
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
        img_dir='train/image',
        ann_dir='train/mask',
        img_suffix='.png',
        seg_map_suffix='_mask.png',  # BreastUltrasoundDataset will handle naming rules
        pipeline=train_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/mask',
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        pipeline=val_pipeline,  # Use val_pipeline (does not contain LoadAnnotations)
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/image',
        ann_dir='test/mask',
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        pipeline=test_pipeline,
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]]))

