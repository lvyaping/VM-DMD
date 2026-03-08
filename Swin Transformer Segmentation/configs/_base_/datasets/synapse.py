# dataset settings for Synapse dataset
dataset_type = 'SynapseDataset'
data_root = '/home/lvyp/Synapse/'

# Image normalization (using ImageNet stats for pre-trained models)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Input size for Synapse (224x224 as in VM-UNet)
img_scale = (224, 224)
crop_size = (224, 224)

# Training pipeline with data augmentation (similar to VM-UNet's RandomGenerator)
# Note: LoadSynapseData is not needed as SynapseDataset already loads data
train_pipeline = [
    dict(type='SynapseRandomAug',  # Random rotation/flip and resize
         output_size=img_scale,
         prob_rot_flip=0.5,
         prob_rotate=0.5,
         rotate_range=(-20, 20)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Validation pipeline (for test_vol data)
# Note: For 3D volumes, we need to process slice by slice
# Wrap in MultiScaleFlipAug to ensure data format matches forward_test expectations
val_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Test pipeline (same as validation)
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        list_dir=f'{data_root}lists/lists_Synapse/',
        train_npz_dir='train_npz',
        test_vol_h5_dir='test_vol_h5',
        pipeline=train_pipeline,
        classes=('background', 'aorta', 'gallbladder', 'spleen', 'left_kidney',
                 'right_kidney', 'liver', 'pancreas', 'stomach'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0]]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='test_vol',  # Use test_vol as validation set
        list_dir=f'{data_root}lists/lists_Synapse/',
        train_npz_dir='train_npz',
        test_vol_h5_dir='test_vol_h5',
        pipeline=val_pipeline,
        test_mode=True,  # Don't load annotations during validation
        classes=('background', 'aorta', 'gallbladder', 'spleen', 'left_kidney',
                 'right_kidney', 'liver', 'pancreas', 'stomach'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0]]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test_vol',
        list_dir=f'{data_root}lists/lists_Synapse/',
        train_npz_dir='train_npz',
        test_vol_h5_dir='test_vol_h5',
        pipeline=test_pipeline,
        classes=('background', 'aorta', 'gallbladder', 'spleen', 'left_kidney',
                 'right_kidney', 'liver', 'pancreas', 'stomach'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0]]))

