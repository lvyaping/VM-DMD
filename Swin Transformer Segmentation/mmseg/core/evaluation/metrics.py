import mmcv
import numpy as np


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


def total_iou_and_dice(results,
                       gt_seg_maps,
                       target_class=1,
                       ignore_index=255,
                       label_map=dict(),
                       reduce_zero_label=False):
    """Calculate total IoU and Dice for binary segmentation (like SegMamba).
    
    This function computes IoU and Dice for a specific class (typically the 
    foreground/lesion class) across all images, similar to SegMamba's approach.
    
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): List of ground truth segmentation maps.
        target_class (int): The class index to compute metrics for (default: 1 for lesion).
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
    
    Returns:
        float: Total IoU for the target class.
        float: Total Dice for the target class.
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i in range(num_imgs):
        pred = results[i]
        gt = gt_seg_maps[i]
        
        # Apply label mapping if provided
        if label_map is not None:
            for old_id, new_id in label_map.items():
                gt[gt == old_id] = new_id
        
        if reduce_zero_label:
            gt[gt == 0] = 255
            gt = gt - 1
            gt[gt == 254] = 255
        
        # Create mask for valid pixels (not ignored)
        valid_mask = (gt != ignore_index)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        # Calculate TP, FP, FN for target class
        # TP: predicted as target_class and ground truth is target_class
        tp = np.sum((pred_valid == target_class) & (gt_valid == target_class))
        # FP: predicted as target_class but ground truth is not target_class
        fp = np.sum((pred_valid == target_class) & (gt_valid != target_class))
        # FN: ground truth is target_class but predicted is not target_class
        fn = np.sum((pred_valid != target_class) & (gt_valid == target_class))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate IoU: TP / (TP + FP + FN)
    total_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    
    # Calculate Dice: 2*TP / (2*TP + FP + FN)
    total_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)
    
    return total_iou, total_dice


def total_precision_recall_f1(results,
                              gt_seg_maps,
                              target_class=1,
                              ignore_index=255,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate total Precision/Recall/F1 for a target class across dataset."""
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for i in range(num_imgs):
        pred = results[i]
        gt = gt_seg_maps[i]
        if label_map is not None:
            for old_id, new_id in label_map.items():
                gt[gt == old_id] = new_id
        if reduce_zero_label:
            gt[gt == 0] = 255
            gt = gt - 1
            gt[gt == 254] = 255
        valid_mask = (gt != ignore_index)
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        tp = np.sum((pred_valid == target_class) & (gt_valid == target_class))
        fp = np.sum((pred_valid == target_class) & (gt_valid != target_class))
        fn = np.sum((pred_valid != target_class) & (gt_valid == target_class))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def total_confusion_binary(results,
                           gt_seg_maps,
                           target_class=1,
                           ignore_index=255,
                           label_map=dict(),
                           reduce_zero_label=False):
    """Accumulate TP, FP, FN, TN for a binary target_class across dataset."""
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    tp = fp = fn = tn = 0
    for i in range(num_imgs):
        pred = results[i]
        gt = gt_seg_maps[i]
        if label_map is not None:
            for old_id, new_id in label_map.items():
                gt[gt == old_id] = new_id
        if reduce_zero_label:
            gt[gt == 0] = 255
            gt = gt - 1
            gt[gt == 254] = 255
        valid = (gt != ignore_index)
        pred_v = pred[valid]
        gt_v = gt[valid]
        tp += np.sum((pred_v == target_class) & (gt_v == target_class))
        fp += np.sum((pred_v == target_class) & (gt_v != target_class))
        fn += np.sum((pred_v != target_class) & (gt_v == target_class))
        tn += np.sum((pred_v != target_class) & (gt_v != target_class))
    return tp, fp, fn, tn
