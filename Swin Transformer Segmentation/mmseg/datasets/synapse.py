import os.path as osp
import numpy as np
import h5py
from scipy.ndimage import zoom
import torch

import mmcv
from mmcv.utils import print_log
from .builder import DATASETS
from .custom import CustomDataset

try:
    from medpy import metric as medpy_metric
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False
    medpy_metric = None
    print("Warning: medpy is not installed. HD95 calculation will be disabled.")
    print("Install with: pip install medpy")


@DATASETS.register_module()
class SynapseDataset(CustomDataset):
    """Synapse dataset for multi-organ segmentation.

    The dataset structure should be:
    .. code-block:: none
        ├── data
        │   ├── Synapse
        │   │   ├── train_npz/
        │   │   │   ├── case0000_slice000.npz
        │   │   │   ├── case0000_slice001.npz
        │   │   │   └── ...
        │   │   ├── test_vol_h5/
        │   │   │   ├── case0000.npy.h5
        │   │   │   ├── case0001.npy.h5
        │   │   │   └── ...
        │   │   └── lists/
        │   │       └── lists_Synapse/
        │   │           ├── train.txt
        │   │           └── test_vol.txt

    For training, data is loaded from .npz files (2D slices).
    For testing, data is loaded from .h5 files (3D volumes).

    Args:
        data_root (str): Root path of the dataset.
        split (str): Split type, 'train' or 'test_vol'.
        list_dir (str): Directory containing split txt files.
        train_npz_dir (str): Directory containing training .npz files.
        test_vol_h5_dir (str): Directory containing test .h5 files.
        pipeline (list[dict]): Processing pipeline.
        test_mode (bool): If True, load test data from .h5 files.
        classes (tuple): Class names. Default: 9 classes (background + 8 organs).
        palette (list): Color palette for visualization.
    """

    # Note: Class order has been adjusted to match actual data based on validation
    # Validation results: All classes conform to medical knowledge characteristics (see VM-UNet/validate_proposed_order.py)
    CLASSES = ('background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
               'liver', 'pancreas', 'spleen', 'stomach')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
               [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
               [64, 0, 0]]

    def __init__(self,
                 data_root,
                 split='train',
                 list_dir=None,
                 train_npz_dir='train_npz',
                 test_vol_h5_dir='test_vol_h5',
                 pipeline=None,
                 test_mode=False,
                 classes=None,
                 palette=None,
                 **kwargs):
        self.split = split
        self.list_dir = list_dir if list_dir else osp.join(data_root, 'lists', 'lists_Synapse')
        self.train_npz_dir = train_npz_dir
        self.test_vol_h5_dir = test_vol_h5_dir
        self.data_root = data_root
        
        # Load sample list from txt file
        split_file = osp.join(self.list_dir, f'{split}.txt')
        if not osp.exists(split_file):
            raise FileNotFoundError(f'Split file not found: {split_file}')
        
        with open(split_file, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        
        # Set data directory based on split
        if split == 'train':
            self.data_dir = osp.join(data_root, train_npz_dir)
        else:  # test_vol
            self.data_dir = osp.join(data_root, test_vol_h5_dir)
        
        # Initialize img_infos BEFORE calling parent __init__
        # (parent will call load_annotations which we override)
        self.img_infos = []
        for sample_name in self.sample_list:
            self.img_infos.append({
                'filename': sample_name,
                'ann': {'seg_map': sample_name}  # For compatibility
            })
        
        # Initialize parent class with minimal required parameters
        # We override most methods, so we pass dummy values
        # Pass split=None to prevent parent from loading annotations
        super(SynapseDataset, self).__init__(
            pipeline=pipeline or [],
            img_dir='',  # Not used for Synapse
            ann_dir='',  # Not used for Synapse
            split=None,  # Don't use parent's split loading
            test_mode=test_mode,
            classes=classes,
            palette=palette,
            **kwargs)
        # Restore split after parent __init__ (parent may have overwritten it)
        self.split = split

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Override parent's load_annotations to avoid scanning empty directories.
        
        We already loaded the sample list in __init__, so just return img_infos.
        """
        return self.img_infos

    def __len__(self):
        return len(self.sample_list)

    def load_data(self, idx):
        """Load image and label from file."""
        sample_name = self.sample_list[idx]
        
        # Determine file extension based on data_dir path
        # CRITICAL: Always use .npz for train_npz directory, .npy.h5 for test_vol_h5 directory
        data_dir_str = str(self.data_dir)
        
        # Force check based on data_dir path
        if 'train_npz' in data_dir_str:
            # Train data: always use .npz
            data_path = osp.join(self.data_dir, f'{sample_name}.npz')
            if not osp.exists(data_path):
                raise FileNotFoundError(
                    f'Train data file not found: {data_path}\n'
                    f'  split={self.split}, data_dir={self.data_dir}\n'
                    f'  sample_name={sample_name}'
                )
            data = np.load(data_path)
            image = data['image']  # Shape: (H, W) or (H, W, C)
            label = data['label']  # Shape: (H, W)
        elif 'test_vol_h5' in data_dir_str:
            # Test data: always use .npy.h5
            data_path = osp.join(self.data_dir, f'{sample_name}.npy.h5')
            if not osp.exists(data_path):
                raise FileNotFoundError(
                    f'Test data file not found: {data_path}\n'
                    f'  split={self.split}, data_dir={self.data_dir}\n'
                    f'  sample_name={sample_name}'
                )
            with h5py.File(data_path, 'r') as f:
                image = f['image'][:]  # Shape: (D, H, W) or (D, H, W, C)
                label = f['label'][:]  # Shape: (D, H, W)
        else:
            # Fallback: use split to determine
            if self.split == 'train':
                data_path = osp.join(self.data_dir, f'{sample_name}.npz')
            else:
                data_path = osp.join(self.data_dir, f'{sample_name}.npy.h5')
            
            if not osp.exists(data_path):
                raise FileNotFoundError(
                    f'Data file not found: {data_path}\n'
                    f'  split={self.split}, data_dir={self.data_dir}\n'
                    f'  sample_name={sample_name}'
                )
            
            if self.split == 'train':
                data = np.load(data_path)
                image = data['image']
                label = data['label']
            else:
                with h5py.File(data_path, 'r') as f:
                    image = f['image'][:]
                    label = f['label'][:]
        
        return image, label, sample_name

    def __getitem__(self, idx):
        """Get training/test data from files."""
        image, label, sample_name = self.load_data(idx)
        
        # Ensure image is 3-channel for pipeline compatibility
        # Synapse images are grayscale (H, W), need to convert to (H, W, 3)
        if len(image.shape) == 2:
            # 2D grayscale: (H, W) -> (H, W, 3)
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] != 3:
            # 3D volume: (D, H, W) -> take first slice and convert to (H, W, 3)
            image = image[0]  # Take first slice
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            # Already 3-channel: (H, W, 3)
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure image is float32 and in [0, 255] range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.float32)
        else:
            image = image.astype(np.float32)
        
        # Ensure label is int64
        label = label.astype(np.int64)
        
        # Prepare data dict for pipeline
        results = {
            'filename': sample_name,
            'ori_filename': sample_name,
            'img': image,
            'img_shape': image.shape[:2],
            'ori_shape': image.shape[:2],
            'pad_shape': image.shape[:2],
            'scale_factor': np.array([1.0, 1.0], dtype=np.float32),
            'flip': False,
            'flip_direction': 'horizontal',  # Default value, may be updated by transforms
            'case_name': sample_name,  # For Synapse-specific use
        }
        
        # Only add gt_semantic_seg if not in test_mode
        if not self.test_mode:
            results['gt_semantic_seg'] = label
        
        # Apply pipeline
        results = self.pipeline(results)
        
        return results

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for idx in range(len(self)):
            _, label, _ = self.load_data(idx)
            if efficient_test:
                # Return file path instead of loading data
                sample_name = self.sample_list[idx]
                if self.split == 'train':
                    data_path = osp.join(self.data_dir, f'{sample_name}.npz')
                else:
                    data_path = osp.join(self.data_dir, f'{sample_name}.npy.h5')
                gt_seg_maps.append(data_path)
            else:
                # Ensure label has the same dimensionality as prediction
                # If label is 3D (from test_vol), take first slice to match __getitem__
                label = np.asarray(label)
                if len(label.shape) == 3 and label.shape[-1] != 3:
                    # 3D volume: (D, H, W) -> take first slice to match __getitem__
                    label = label[0]  # Shape: (H, W)
                elif len(label.shape) == 2:
                    # Already 2D: (H, W)
                    pass
                else:
                    # Unexpected shape, try to handle
                    if len(label.shape) == 3 and label.shape[-1] == 3:
                        # (H, W, 3) - this shouldn't happen for labels, but handle it
                        label = label[:, :, 0]  # Take first channel
                gt_seg_maps.append(label)
        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        For Synapse dataset, we compute Dice and HD95 for each class.
        
        Args:
            results: Prediction results from model inference.
            metric: Metrics to compute ('mIoU', 'mDice', 'mHD95').
            logger: Logger object.
            **kwargs: Additional arguments:
                - use_3d_eval (bool): If True, use 3D volume evaluation (requires model).
                - model: Model object (required if use_3d_eval=True).
                - img_norm_cfg: Image normalization config (optional).
                - patch_size: Patch size for 3D evaluation (default: (224, 224)).
        """
        # Check if 3D evaluation is requested
        use_3d_eval = kwargs.get('use_3d_eval', False)
        # Debug: log the evaluation mode
        if logger is not None:
            logger.info(f'Evaluation mode check: use_3d_eval={use_3d_eval}, split={self.split}, type(use_3d_eval)={type(use_3d_eval)}')
        if use_3d_eval and self.split == 'test_vol':
            if logger is not None:
                logger.info('Using 3D volume evaluation mode')
            return self._evaluate_3d(results, metric=metric, logger=logger, **kwargs)
        else:
            if logger is not None:
                logger.info(f'Using 2D slice evaluation mode (use_3d_eval={use_3d_eval}, split={self.split})')
        
        # Standard 2D evaluation (current implementation)
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mHD95']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError(f'metric {metric} is not supported')
        
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        
        # Convert results to numpy arrays
        if isinstance(results[0], np.ndarray):
            pred_seg_maps = results
        else:
            # Results might be in different format, convert them
            pred_seg_maps = []
            for result in results:
                if isinstance(result, dict):
                    pred_seg_maps.append(result['seg_pred'])
                else:
                    pred_seg_maps.append(result)
        
        # Ensure all predictions and ground truth are numpy arrays
        pred_seg_maps = [np.asarray(pred) for pred in pred_seg_maps]
        gt_seg_maps = [np.asarray(gt) for gt in gt_seg_maps]
        
        # Compute metrics for each class (skip background class 0)
        num_classes = len(self.CLASSES)
        dice_scores = []
        hd95_scores = []
        
        for class_idx in range(1, num_classes):  # Skip background
            class_dice = []
            class_hd95 = []
            
            for pred, gt in zip(pred_seg_maps, gt_seg_maps):
                # Ensure pred and gt are numpy arrays
                pred = np.asarray(pred)
                gt = np.asarray(gt)
                
                # Convert to binary masks for this class
                pred_binary = (pred == class_idx).astype(np.uint8)
                gt_binary = (gt == class_idx).astype(np.uint8)
                
                # Ensure binary masks have the same shape
                if pred_binary.shape != gt_binary.shape:
                    # Resize to match if shapes differ
                    from scipy.ndimage import zoom
                    if len(pred_binary.shape) == len(gt_binary.shape):
                        # Same dimensionality, resize pred to match gt
                        zoom_factors = [gt_binary.shape[i] / pred_binary.shape[i] 
                                      for i in range(len(pred_binary.shape))]
                        pred_binary = zoom(pred_binary, zoom_factors, order=0).astype(np.uint8)
                    else:
                        # Different dimensionality, skip HD95 for this case
                        if logger is not None:
                            logger.warning(f'HD95 skipped for class {class_idx}: shape mismatch '
                                         f'(pred: {pred_binary.shape}, gt: {gt_binary.shape})')
                        hd95 = 0.0
                        class_hd95.append(hd95)
                        # Still compute Dice
                        intersection = np.logical_and(pred_binary, gt_binary).sum()
                        union = pred_binary.sum() + gt_binary.sum()
                        if union > 0:
                            dice = 2.0 * intersection / union
                        else:
                            dice = 1.0 if union == 0 else 0.0
                        class_dice.append(dice)
                        continue
                
                # Compute Dice
                intersection = np.logical_and(pred_binary, gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum()
                if union > 0:
                    dice = 2.0 * intersection / union
                else:
                    dice = 1.0 if union == 0 else 0.0
                class_dice.append(dice)
                
                # Compute HD95 using medpy (same as VM-UNet)
                if HAS_MEDPY and medpy_metric is not None:
                    try:
                        # Ensure binary masks are numpy arrays with same shape
                        pred_binary = np.asarray(pred_binary, dtype=np.uint8)
                        gt_binary = np.asarray(gt_binary, dtype=np.uint8)
                        
                        # Verify shapes match
                        if pred_binary.shape != gt_binary.shape:
                            if logger is not None:
                                logger.warning(f'HD95 skipped for class {class_idx}: shape mismatch '
                                             f'(pred: {pred_binary.shape}, gt: {gt_binary.shape})')
                            hd95 = 0.0
                        elif pred_binary.sum() > 0 and gt_binary.sum() > 0:
                            # medpy.metric.binary.hd95 expects binary masks as numpy arrays with same shape
                            hd95 = medpy_metric.binary.hd95(pred_binary, gt_binary)
                        elif pred_binary.sum() > 0 and gt_binary.sum() == 0:
                            # Prediction exists but ground truth doesn't
                            hd95 = 0.0  # Or could use a large penalty value
                        else:
                            # Both are empty
                            hd95 = 0.0
                    except Exception as e:
                        # If HD95 calculation fails, use 0 as fallback
                        if logger is not None:
                            logger.warning(f'HD95 calculation failed for class {class_idx}: {e}')
                        hd95 = 0.0
                else:
                    hd95 = 0.0  # medpy not available
                
                class_hd95.append(hd95)
            
            dice_scores.append(np.mean(class_dice))
            hd95_scores.append(np.mean(class_hd95))
        
        # Compute mean Dice and mean HD95
        mean_dice = np.mean(dice_scores)
        mean_hd95 = np.mean(hd95_scores)
        
        eval_results['mDice'] = mean_dice
        eval_results['mHD95'] = mean_hd95
        
        # Log per-class results
        if logger is not None:
            logger.info('=' * 50)
            logger.info('Synapse Dataset Evaluation Results:')
            logger.info(f'Mean Dice: {mean_dice:.4f}')
            logger.info(f'Mean HD95: {mean_hd95:.4f}')
            logger.info('Per-class Metrics:')
            for i, class_name in enumerate(self.CLASSES[1:], 1):
                logger.info(f'  {class_name}: Dice={dice_scores[i-1]:.4f}, HD95={hd95_scores[i-1]:.4f}')
            logger.info('=' * 50)
        else:
            print('=' * 50)
            print('Synapse Dataset Evaluation Results:')
            print(f'Mean Dice: {mean_dice:.4f}')
            print(f'Mean HD95: {mean_hd95:.4f}')
            print('Per-class Metrics:')
            for i, class_name in enumerate(self.CLASSES[1:], 1):
                print(f'  {class_name}: Dice={dice_scores[i-1]:.4f}, HD95={hd95_scores[i-1]:.4f}')
            print('=' * 50)
        
        return eval_results
    
    def _evaluate_3d(self, results, metric='mIoU', logger=None, **kwargs):
        """3D volume evaluation using test_single_volume_3d.
        
        This method loads full 3D volumes and evaluates them slice-by-slice,
        then computes metrics on the complete 3D volumes.
        """
        from .synapse_3d_eval import test_single_volume_3d
        
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mHD95']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError(f'metric {metric} is not supported')
        
        # Get model from kwargs
        model = kwargs.get('model', None)
        if model is None:
            raise ValueError('model is required for 3D evaluation. Pass it via kwargs: evaluate(results, model=model, use_3d_eval=True)')
        
        # Get configuration
        img_norm_cfg = kwargs.get('img_norm_cfg', None)
        patch_size = kwargs.get('patch_size', (224, 224))
        device = kwargs.get('device', 'cuda')
        
        # Get device from model
        if hasattr(model, 'module'):
            model_device = next(model.module.parameters()).device
        else:
            model_device = next(model.parameters()).device
        device = str(model_device)
        
        # Get image normalization config from model cfg if not provided
        if img_norm_cfg is None and hasattr(model, 'cfg'):
            img_norm_cfg = model.cfg.get('img_norm_cfg', None)
        
        num_classes = len(self.CLASSES)
        
        # Evaluate each 3D volume
        all_metric_lists = []
        
        if logger is not None:
            logger.info('Starting 3D volume evaluation...')
            logger.info(f'Number of volumes: {len(self.sample_list)}')
        else:
            print('Starting 3D volume evaluation...')
            print(f'Number of volumes: {len(self.sample_list)}')
        
        for idx in range(len(self.sample_list)):
            sample_name = self.sample_list[idx]
            
            # Load full 3D volume
            image, label, _ = self.load_data(idx)
            
            # Ensure image and label are 3D
            if len(image.shape) == 2:
                # Single slice, convert to 3D
                image = image[np.newaxis, :, :]
            if len(label.shape) == 2:
                label = label[np.newaxis, :, :]
            
            # Evaluate this volume
            try:
                prediction, metric_list = test_single_volume_3d(
                    image, label, model, num_classes,
                    patch_size=patch_size,
                    device=device,
                    img_norm_cfg=img_norm_cfg,
                    logger=logger
                )
                all_metric_lists.append(metric_list)
                
                if logger is not None:
                    logger.info(f'Volume {idx+1}/{len(self.sample_list)} ({sample_name}): '
                              f'Mean Dice={np.mean([m[0] for m in metric_list]):.4f}, '
                              f'Mean HD95={np.mean([m[1] for m in metric_list]):.4f}')
            except Exception as e:
                if logger is not None:
                    logger.error(f'Error evaluating volume {sample_name}: {e}')
                else:
                    print(f'Error evaluating volume {sample_name}: {e}')
                # Use zero metrics for failed volumes
                all_metric_lists.append([(0.0, 0.0) for _ in range(num_classes - 1)])
        
        # Aggregate metrics across all volumes
        # all_metric_lists: list of lists, each inner list has (dice, hd95) for each class
        # Shape: (num_volumes, num_classes-1, 2)
        num_volumes = len(all_metric_lists)
        num_organs = num_classes - 1  # Exclude background
        
        dice_scores = []
        hd95_scores = []
        
        for class_idx in range(num_organs):
            class_dice = [all_metric_lists[v][class_idx][0] for v in range(num_volumes)]
            class_hd95 = [all_metric_lists[v][class_idx][1] for v in range(num_volumes)]
            dice_scores.append(np.mean(class_dice))
            hd95_scores.append(np.mean(class_hd95))
        
        # Compute mean Dice and mean HD95
        mean_dice = np.mean(dice_scores)
        mean_hd95 = np.mean(hd95_scores)
        
        eval_results = {}
        eval_results['mDice'] = mean_dice
        eval_results['mHD95'] = mean_hd95
        
        # Log results
        if logger is not None:
            logger.info('=' * 50)
            logger.info('Synapse Dataset 3D Volume Evaluation Results:')
            logger.info(f'Mean Dice: {mean_dice:.4f}')
            logger.info(f'Mean HD95: {mean_hd95:.4f}')
            logger.info('Per-class Metrics:')
            for i, class_name in enumerate(self.CLASSES[1:], 1):
                logger.info(f'  {class_name}: Dice={dice_scores[i-1]:.4f}, HD95={hd95_scores[i-1]:.4f}')
            logger.info('=' * 50)
        else:
            print('=' * 50)
            print('Synapse Dataset 3D Volume Evaluation Results:')
            print(f'Mean Dice: {mean_dice:.4f}')
            print(f'Mean HD95: {mean_hd95:.4f}')
            print('Per-class Metrics:')
            for i, class_name in enumerate(self.CLASSES[1:], 1):
                print(f'  {class_name}: Dice={dice_scores[i-1]:.4f}, HD95={hd95_scores[i-1]:.4f}')
            print('=' * 50)
        
        return eval_results

