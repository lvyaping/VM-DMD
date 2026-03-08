import os.path as osp

import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from mmseg.core.evaluation.metrics import total_confusion_binary


@DATASETS.register_module()
class ISIC2017Dataset(CustomDataset):
    """ISIC2017 dataset for skin lesion segmentation.

    In segmentation map annotation for ISIC2017, 0 stands for background, 255 stands
    for lesion. We need to convert 255 to 1 for training and evaluation.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '_segmentation.png'.
    """

    CLASSES = ('background', 'lesion')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        # Get parameters from kwargs, use defaults if not present
        # This avoids conflicts with parameters in config files
        img_suffix = kwargs.pop('img_suffix', '.jpg')
        seg_map_suffix = kwargs.pop('seg_map_suffix', '_segmentation.png')
        reduce_zero_label = kwargs.pop('reduce_zero_label', False)
        
        super(ISIC2017Dataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert osp.exists(self.img_dir)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation.
        
        Override this method to convert label values from 0/255 to 0/1.
        """
        gt_seg_maps = []
        for img_info in self.img_infos:
            # Get segmentation map path
            # img_info should have 'ann' key if ann_dir is provided
            if 'ann' in img_info and 'seg_map' in img_info['ann']:
                seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            else:
                # Fallback: construct seg_map from img filename
                img_filename = img_info['filename']
                seg_map = img_filename.replace('.jpg', '_segmentation.png')
                if self.ann_dir:
                    seg_map = osp.join(self.ann_dir, osp.basename(seg_map))
            
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
                # Convert label values: 255 -> 1
                if isinstance(gt_seg_map, np.ndarray):
                    converted_map = gt_seg_map.copy()
                    converted_map[gt_seg_map == 255] = 1
                    gt_seg_map = converted_map
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset with total IoU and Dice (like SegMamba).
        
        This method extends the parent evaluate method to also compute and 
        display total IoU and Dice for the lesion class (class 1), similar 
        to SegMamba's evaluation approach.
        """
        # Call parent evaluate method to get standard metrics
        # VM-UNet style: only report binary lesion class IoU (miou), F1/Dice (f1_or_dsc) and overall accuracy
        # Do not return general metrics like mIoU/mAcc/aAcc to ensure alignment for comparison
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
        
        eval_results['miou'] = miou
        eval_results['f1_or_dsc'] = f1_or_dsc
        eval_results['accuracy'] = accuracy
        eval_results['sensitivity'] = sensitivity
        eval_results['specificity'] = specificity
        
        # Print all metrics in table format
        # Format: Miou | Dsc | Acc | Spe | Sen
        header = f"{'Miou':<10} {'Dsc':<10} {'Acc':<10} {'Spe':<10} {'Sen':<10}"
        values = f"{miou:<10.4f} {f1_or_dsc:<10.4f} {accuracy:<10.4f} {specificity:<10.4f} {sensitivity:<10.4f}"
        
        if logger is not None:
            logger.info('=' * 70)
            logger.info('ISIC2017 Evaluation Metrics (Lesion, dataset-level):')
            logger.info('-' * 70)
            logger.info(header)
            logger.info(values)
            logger.info('=' * 70)
        else:
            print('=' * 70)
            print('ISIC2017 Evaluation Metrics (Lesion, dataset-level):')
            print('-' * 70)
            print(header)
            print(values)
            print('=' * 70)
        
        return eval_results

