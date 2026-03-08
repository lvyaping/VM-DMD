import os.path as osp
import re

import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from mmseg.core.evaluation.metrics import total_confusion_binary


@DATASETS.register_module()
class BreastUltrasoundDataset(CustomDataset):
    """Breast Ultrasound dataset for lesion segmentation.

    In segmentation map annotation for Breast Ultrasound, 0 stands for background, 1 stands
    for lesion. Labels are already in 0/1 format, no conversion needed.

    The file naming rules supported:
    1. bus_0001-l.png -> mask_0001-l.png (replace 'bus_' with 'mask_')
    2. benign_02.png -> benign_mask_02.png (insert '_mask' before number)
    
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_mask.png', but we need to handle the special naming patterns.
    """

    CLASSES = ('background', 'lesion')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        # Get parameters from kwargs, use defaults if not present
        img_suffix = kwargs.pop('img_suffix', '.png')
        seg_map_suffix = kwargs.pop('seg_map_suffix', '_mask.png')
        reduce_zero_label = kwargs.pop('reduce_zero_label', False)
        
        super(BreastUltrasoundDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert osp.exists(self.img_dir)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Override to handle special naming pattern: benign_02.png -> benign_mask_02.png
        
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file.

        Returns:
            list[dict]: All image info of dataset.
        """
        from mmcv.utils import print_log
        from mmseg.utils import get_root_logger

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        # Handle naming pattern: benign_02 -> benign_mask_02
                        seg_map = self._get_mask_filename(img_name + img_suffix, img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    # Handle naming pattern: benign_02.png -> benign_mask_02.png
                    seg_map = self._get_mask_filename(img, img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def _get_mask_filename(self, img_filename, img_suffix, seg_map_suffix):
        """Convert image filename to mask filename.
        
        Handles multiple naming patterns:
        1. bus_0001-l.png -> mask_0001-l.png (replace 'bus_' with 'mask_')
        2. benign_02.png -> benign_mask_02.png (insert '_mask' before number)
        
        Args:
            img_filename (str): Image filename (e.g., 'bus_0001-l.png' or 'benign_02.png')
            img_suffix (str): Image suffix (e.g., '.png')
            seg_map_suffix (str): Mask suffix (e.g., '_mask.png' or '.png')
        
        Returns:
            str: Mask filename (e.g., 'mask_0001-l.png' or 'benign_mask_02.png')
        """
        # Remove img_suffix to get base name
        base_name = img_filename.replace(img_suffix, '')
        
        # Pattern 1: bus_XXXX-suffix.png -> mask_XXXX-suffix.png
        # This handles the Breast_BUS dataset naming convention
        if base_name.startswith('bus_'):
            # Replace 'bus_' with 'mask_' and keep the rest
            mask_base = base_name.replace('bus_', 'mask_', 1)
            # If seg_map_suffix is '_mask.png', replace it with '.png'
            if seg_map_suffix == '_mask.png':
                mask_filename = mask_base + '.png'
            else:
                mask_filename = mask_base + seg_map_suffix
        # Pattern 2: prefix_number.png -> prefix_mask_number.png
        # This handles the original benign/malignant naming convention
        else:
            match = re.match(r'(.+?)_(\d+)$', base_name)
            if match:
                prefix = match.group(1)  # e.g., 'benign' or 'malignant'
                number = match.group(2)   # e.g., '02' or '195'
                # If seg_map_suffix contains '_mask', extract only the suffix part (e.g., '.png')
                # Otherwise use seg_map_suffix as is
                if seg_map_suffix.startswith('_mask'):
                    # Extract the suffix part (e.g., '_mask.png' -> '.png')
                    actual_suffix = seg_map_suffix.replace('_mask', '', 1)
                    mask_filename = f"{prefix}_mask_{number}{actual_suffix}"
                else:
                    # seg_map_suffix is just the suffix (e.g., '.png')
                    mask_filename = f"{prefix}_mask_{number}{seg_map_suffix}"
            else:
                # Fallback: use standard replacement
                mask_filename = base_name + seg_map_suffix
        
        return mask_filename

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation.
        
        Labels are already in 0/1 format, no conversion needed.
        """
        gt_seg_maps = []
        for img_info in self.img_infos:
            # Get segmentation map path
            if 'ann' in img_info and 'seg_map' in img_info['ann']:
                seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            else:
                # Fallback: construct seg_map from img filename
                img_filename = img_info['filename']
                seg_map = self._get_mask_filename(
                    img_filename, 
                    self.img_suffix, 
                    self.seg_map_suffix
                )
                if self.ann_dir:
                    seg_map = osp.join(self.ann_dir, seg_map)
            
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
                # Labels are already in 0/1 format, no conversion needed
                # But ensure values are in [0, 1]
                if isinstance(gt_seg_map, np.ndarray):
                    # Clamp values to [0, 1] in case there are any other values
                    gt_seg_map = np.clip(gt_seg_map, 0, 1).astype(np.uint8)
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset with total IoU and Dice.
        
        This method extends the parent evaluate method to also compute and 
        display total IoU and Dice for the lesion class (class 1).
        """
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        tp, fp, fn, tn = total_confusion_binary(
            results=results,
            gt_seg_maps=gt_seg_maps,
            target_class=1,
            ignore_index=self.ignore_index,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label
        )
        # Calculate all metrics
        denom_iou = (tp + fp + fn) + 1e-8
        miou = float(tp) / float(denom_iou)
        denom_dice = (2 * tp + fp + fn) + 1e-8
        f1_or_dsc = float(2 * tp) / float(denom_dice)
        total = tp + fp + fn + tn + 1e-8
        accuracy = float(tp + tn) / float(total)
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = float(tp) / float(tp + fn + 1e-8)
        # Specificity = TN / (TN + FP)
        specificity = float(tn) / float(tn + fp + 1e-8)
        # Precision = TP / (TP + FP)
        precision = float(tp) / float(tp + fp + 1e-8)
        
        eval_results['miou'] = miou
        eval_results['f1_or_dsc'] = f1_or_dsc
        eval_results['accuracy'] = accuracy
        eval_results['sensitivity'] = sensitivity
        eval_results['specificity'] = specificity
        eval_results['precision'] = precision
        
        # Print all metrics in table format
        header = f"{'Miou':<10} {'Dsc':<10} {'Acc':<10} {'Spe':<10} {'Sen':<10} {'Pre':<10}"
        values = f"{miou:<10.4f} {f1_or_dsc:<10.4f} {accuracy:<10.4f} {specificity:<10.4f} {sensitivity:<10.4f} {precision:<10.4f}"
        
        if logger is not None:
            logger.info('=' * 70)
            logger.info('Breast Ultrasound Evaluation Metrics (Lesion, dataset-level):')
            logger.info('-' * 70)
            logger.info(header)
            logger.info(values)
            logger.info('=' * 70)
        else:
            print('=' * 70)
            print('Breast Ultrasound Evaluation Metrics (Lesion, dataset-level):')
            print('-' * 70)
            print(header)
            print(values)
            print('=' * 70)
        
        return eval_results

