import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import zoom
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadSynapseData(object):
    """Load Synapse data from .npz or .h5 files.
    
    This transform expects that the data has already been loaded by SynapseDataset
    and is available in results['img'] and results['gt_semantic_seg'].
    It mainly ensures the data format is correct.
    
    Args:
        split (str): 'train' or 'test_vol'. Default: 'train'.
    """
    
    def __init__(self, split='train'):
        self.split = split
    
    def __call__(self, results):
        """Process the data."""
        img = results['img']
        gt_semantic_seg = results.get('gt_semantic_seg', None)
        
        # Ensure image is in correct format
        # For Synapse, images are grayscale (H, W) or (D, H, W)
        # We need to convert to RGB format (H, W, 3) or (D, H, W, 3)
        if len(img.shape) == 2:
            # 2D image: (H, W) -> (H, W, 3)
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[-1] != 3:
            # 3D volume: (D, H, W) -> (D, H, W, 3)
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[-1] == 3:
            # Already RGB: (H, W, 3)
            pass
        else:
            # 4D: (D, H, W, 3) - already correct
            pass
        
        # Normalize image to [0, 255] if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        results['img'] = img.astype(np.float32)
        
        if gt_semantic_seg is not None:
            results['gt_semantic_seg'] = gt_semantic_seg.astype(np.int64)
        
        return results


def random_rot_flip(image, label):
    """Random rotation (90 degrees) and flip."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, angle_range=(-20, 20)):
    """Random rotation with small angle."""
    angle = np.random.randint(angle_range[0], angle_range[1])
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


@PIPELINES.register_module()
class SynapseRandomAug(object):
    """Random augmentation for Synapse dataset.
    
    Similar to VM-UNet's RandomGenerator, includes:
    - Random rotation (90 degrees) + flip
    - Random small angle rotation
    - Resize to output_size
    
    Args:
        output_size (tuple): Target output size (H, W).
        prob_rot_flip (float): Probability of applying rotation+flip. Default: 0.5.
        prob_rotate (float): Probability of applying small rotation. Default: 0.5.
        rotate_range (tuple): Range of rotation angles. Default: (-20, 20).
    """
    
    def __init__(self,
                 output_size=(224, 224),
                 prob_rot_flip=0.5,
                 prob_rotate=0.5,
                 rotate_range=(-20, 20)):
        self.output_size = output_size
        self.prob_rot_flip = prob_rot_flip
        self.prob_rotate = prob_rotate
        self.rotate_range = rotate_range
    
    def __call__(self, results):
        """Apply random augmentation."""
        img = results['img']
        gt_semantic_seg = results.get('gt_semantic_seg', None)
        
        # Handle 2D or 3D images
        # CRITICAL: Ensure we always work with 2D RGB images (H, W, 3)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            # 2D RGB image: (H, W, 3)
            image_2d = img
        elif len(img.shape) == 4:
            # 3D volume: (D, H, W, 3) - process first slice for now
            image_2d = img[0]  # Take first slice
        elif len(img.shape) == 2:
            # 2D grayscale: (H, W) -> convert to (H, W, 3)
            image_2d = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[-1] != 3:
            # 3D grayscale: (D, H, W) -> take first slice and convert to (H, W, 3)
            image_2d = img[0]
            image_2d = np.stack([image_2d, image_2d, image_2d], axis=-1)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # Ensure image_2d is (H, W, 3)
        if len(image_2d.shape) != 3 or image_2d.shape[-1] != 3:
            raise ValueError(f"After processing, image_2d should be (H, W, 3), got {image_2d.shape}")
        
        # Get label
        if gt_semantic_seg is not None:
            if len(gt_semantic_seg.shape) == 2:
                label_2d = gt_semantic_seg
            elif len(gt_semantic_seg.shape) == 3:
                label_2d = gt_semantic_seg[0]  # Take first slice
            else:
                label_2d = gt_semantic_seg
        else:
            label_2d = None
        
        # Apply random augmentation
        if random.random() > (1 - self.prob_rot_flip):
            image_2d, label_2d = random_rot_flip(image_2d, label_2d)
            # Update flip info
            results['flip'] = True
            results['flip_direction'] = 'horizontal'  # random_rot_flip may flip horizontally or vertically
        elif random.random() > (1 - self.prob_rotate):
            if label_2d is not None:
                image_2d, label_2d = random_rotate(image_2d, label_2d, self.rotate_range)
            else:
                image_2d = random_rotate(image_2d, image_2d, self.rotate_range)[0]
            # Rotation doesn't count as flip
            results['flip'] = False
        
        # Resize to output_size
        x, y = image_2d.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            # Resize image (use order=3 for bicubic interpolation)
            # image_2d should always be (H, W, 3) at this point
            image_2d = zoom(image_2d, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            
            # Resize label (use order=0 for nearest neighbor)
            if label_2d is not None:
                label_2d = zoom(label_2d, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Ensure output is still (H, W, 3)
        if len(image_2d.shape) != 3 or image_2d.shape[-1] != 3:
            raise ValueError(f"After resize, image_2d should be (H, W, 3), got {image_2d.shape}")
        
        # Update results - always output (H, W, 3)
        results['img'] = image_2d.astype(np.float32)
        
        if label_2d is not None:
            results['gt_semantic_seg'] = label_2d.astype(np.int64)
        
        # Update shape info
        results['img_shape'] = image_2d.shape[:2]
        results['ori_shape'] = image_2d.shape[:2]
        results['pad_shape'] = image_2d.shape[:2]
        
        return results

