"""
3D volume evaluation function - similar to VM-UNet's test_single_volume
Used to implement full 3D volume support in Swin-Transformer project
"""
import numpy as np
from scipy.ndimage import zoom
import torch

try:
    from medpy import metric as medpy_metric
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False
    medpy_metric = None


def test_single_volume_3d(image, label, model, num_classes, patch_size=(224, 224), 
                          device='cuda', img_norm_cfg=None, logger=None):
    """
    Process a single 3D volume, predict slice-by-slice and combine (similar to VM-UNet's test_single_volume)
    
    Args:
        image: numpy array, shape (D, H, W) or (D, H, W, C)
        label: numpy array, shape (D, H, W)
        model: PyTorch model (MMSegmentation segmentor)
        num_classes: int, number of classes
        patch_size: tuple, model input size (H, W)
        device: str, device ('cuda' or 'cpu')
        img_norm_cfg: dict, image normalization config (mean, std)
        logger: logger object, optional
    
    Returns:
        prediction: numpy array, shape (D, H, W), complete 3D prediction result
        metric_list: list, (dice, hd95) metrics for each class
    """
    # Ensure numpy arrays
    image = np.asarray(image)
    label = np.asarray(label)
    
    # Default normalization config (ImageNet standard)
    if img_norm_cfg is None:
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )
    
    # Process 3D volume
    if len(image.shape) == 3:
        # 3D volume: (D, H, W)
        prediction = np.zeros_like(label, dtype=np.int64)
        
        # Get model device
        if hasattr(model, 'module'):
            model_device = next(model.module.parameters()).device
        else:
            model_device = next(model.parameters()).device
        
        model.eval()
        with torch.no_grad():
            for ind in range(image.shape[0]):
                # Extract single slice
                slice_img = image[ind, :, :]  # (H, W)
                original_shape = slice_img.shape
                
                # Ensure image values are in [0, 255] range
                if slice_img.max() <= 1.0:
                    slice_img = slice_img * 255.0
                slice_img = np.clip(slice_img, 0, 255).astype(np.float32)
                
                # Convert to 3-channel (H, W, 3)
                if len(slice_img.shape) == 2:
                    slice_img = np.stack([slice_img, slice_img, slice_img], axis=-1)
                
                # Resize to model input size (before normalization)
                if slice_img.shape[0] != patch_size[0] or slice_img.shape[1] != patch_size[1]:
                    slice_img = zoom(slice_img, 
                                   (patch_size[0] / slice_img.shape[0], 
                                    patch_size[1] / slice_img.shape[1], 1), 
                                   order=3)
                
                # Normalization config
                if img_norm_cfg is None:
                    # Default ImageNet normalization
                    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                    to_rgb = True
                else:
                    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
                    std = np.array(img_norm_cfg['std'], dtype=np.float32)
                    to_rgb = img_norm_cfg.get('to_rgb', True)
                
                # BGR -> RGB conversion (if needed)
                if to_rgb:
                    slice_img = slice_img[:, :, ::-1]  # BGR -> RGB
                
                # Normalize
                slice_img = (slice_img - mean) / std
                
                # Convert to tensor and add batch dimension
                # MMSegmentation expected format: (1, 3, H, W)
                slice_tensor = torch.from_numpy(slice_img).permute(2, 0, 1).unsqueeze(0).float()
                slice_tensor = slice_tensor.to(model_device)
                
                # Create img_meta (required by MMSegmentation)
                img_meta = [{
                    'ori_shape': (original_shape[0], original_shape[1], 3),
                    'img_shape': (slice_img.shape[0], slice_img.shape[1], 3),
                    'pad_shape': (slice_img.shape[0], slice_img.shape[1], 3),
                    'scale_factor': np.array([slice_img.shape[0] / original_shape[0], 
                                             slice_img.shape[1] / original_shape[1]], dtype=np.float32),
                    'flip': False,
                    'flip_direction': None
                }]
                
                # Predict: use simple_test method
                # Note: if model is wrapped by DataParallel, need to access model.module
                model_to_test = model.module if hasattr(model, 'module') else model
                result = model_to_test.simple_test(slice_tensor, img_meta, rescale=True)
                
                # Extract prediction result
                if isinstance(result, list):
                    pred_slice = result[0]
                else:
                    pred_slice = result
                
                # Convert to numpy
                if isinstance(pred_slice, torch.Tensor):
                    pred_slice = pred_slice.cpu().numpy()
                
                # Ensure 2D array
                if len(pred_slice.shape) == 2:
                    pass  # Already 2D
                elif len(pred_slice.shape) == 3:
                    pred_slice = pred_slice[0]  # Take first
                else:
                    pred_slice = np.squeeze(pred_slice)
                
                # Resize back to original size
                if pred_slice.shape[0] != original_shape[0] or pred_slice.shape[1] != original_shape[1]:
                    zoom_factors = (original_shape[0] / pred_slice.shape[0],
                                  original_shape[1] / pred_slice.shape[1])
                    pred_slice = zoom(pred_slice, zoom_factors, order=0).astype(np.int64)
                else:
                    pred_slice = pred_slice.astype(np.int64)
                
                # Store prediction result
                prediction[ind] = pred_slice
                
    else:
        # 2D image, predict directly
        raise NotImplementedError("2D image prediction not implemented in this function")
    
    # Compute metrics for each class
    metric_list = []
    for class_idx in range(1, num_classes):  # Skip background
        pred_binary = (prediction == class_idx).astype(np.uint8)
        gt_binary = (label == class_idx).astype(np.uint8)
        
        # Compute Dice
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        if union > 0:
            dice = 2.0 * intersection / union
        else:
            dice = 1.0 if union == 0 else 0.0
        
        # Compute HD95
        if HAS_MEDPY and medpy_metric is not None:
            try:
                if pred_binary.sum() > 0 and gt_binary.sum() > 0:
                    hd95 = medpy_metric.binary.hd95(pred_binary, gt_binary)
                else:
                    hd95 = 0.0
            except Exception as e:
                if logger is not None:
                    logger.warning(f'HD95 calculation failed for class {class_idx}: {e}')
                hd95 = 0.0
        else:
            hd95 = 0.0
        
        metric_list.append((dice, hd95))
    
    return prediction, metric_list

