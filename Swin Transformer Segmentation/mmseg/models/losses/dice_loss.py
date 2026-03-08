import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255,
              batch_dice=False):
    """Calculate Dice loss.
    
    Reference: SegMamba/light_training/loss/dice.py
    Uses the same formula: dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    Then loss = 1 - dice (or -dice for SegMamba style)

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, H, W) or (N, C, D, H, W).
        target (torch.Tensor): The ground truth with shape (N, H, W) or (N, D, H, W).
        valid_mask (torch.Tensor): The valid mask with shape (N, H, W) or (N, D, H, W).
        smooth (float): Smoothing factor. Default: 1.
        exponent (float): Exponent factor. Default: 2.
        class_weight (torch.Tensor, optional): Class weight. Default: None.
        ignore_index (int): The label index to be ignored. Default: 255.
        batch_dice (bool): If True, calculate Dice over batch. Default: False.

    Returns:
        torch.Tensor: The calculated Dice loss.
    """
    assert pred.shape[0] == target.shape[0]
    num_classes = pred.shape[1]
    
    # Apply softmax to predictions
    pred = F.softmax(pred, dim=1)
    
    # Convert target to one-hot encoding
    # target shape: (N, H, W) -> (N, 1, H, W)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # Ensure target values are in valid range [0, num_classes-1]
    # Clamp invalid values (like 255 for ignore_index) to 0
    target_long = target.long()
    if ignore_index is not None:
        # Replace ignore_index with 0 (will be masked out later)
        target_long = torch.where(target_long == ignore_index, 
                                  torch.zeros_like(target_long), 
                                  target_long)
    # Clamp to valid range [0, num_classes-1]
    target_long = torch.clamp(target_long, min=0, max=num_classes - 1)
    
    # Create one-hot encoding: (N, C, H, W)
    target_onehot = torch.zeros_like(pred)
    target_onehot.scatter_(1, target_long, 1)
    
    # Apply valid mask if provided
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        pred = pred * valid_mask
        target_onehot = target_onehot * valid_mask
    
    # Calculate axes for reduction
    # For 2D: axes = [2, 3] (H, W)
    # For 3D: axes = [2, 3, 4] (D, H, W)
    axes = list(range(2, pred.dim()))
    
    if batch_dice:
        # Calculate Dice over batch and spatial dimensions
        axes = [0] + axes
    
    # Calculate tp, fp, fn for each class (following SegMamba's approach)
    # tp (true positive) = pred * target_onehot
    # fp (false positive) = pred * (1 - target_onehot)
    # fn (false negative) = (1 - pred) * target_onehot
    tp = (pred * target_onehot).sum(dim=axes)
    fp = (pred * (1 - target_onehot)).sum(dim=axes)
    fn = ((1 - pred) * target_onehot).sum(dim=axes)
    
    # Calculate Dice coefficient for each class
    # Dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    # This is the same as: (2*intersection + smooth) / (pred_sum + target_sum + smooth)
    numerator = 2 * tp + smooth
    denominator = 2 * tp + fp + fn + smooth
    dice = numerator / torch.clamp(denominator, min=1e-8)
    
    # Dice loss = 1 - Dice (mean over classes)
    dice_loss_per_class = 1 - dice
    
    # Apply class weight if provided
    if class_weight is not None:
        if batch_dice:
            dice_loss_per_class = dice_loss_per_class * class_weight
        else:
            # class_weight shape: (C,), dice_loss_per_class shape: (N, C)
            dice_loss_per_class = dice_loss_per_class * class_weight.unsqueeze(0)
    
    # Mean over classes
    if batch_dice:
        # dice_loss_per_class shape: (C,)
        total_loss = dice_loss_per_class.mean()
    else:
        # dice_loss_per_class shape: (N, C)
        # First mean over classes (dim=1), then mean over batch (dim=0)
        # This ensures we get the average Dice loss per sample, then average over batch
        total_loss = dice_loss_per_class.mean(dim=1).mean()
    
    # Ensure loss is in reasonable range [0, 1]
    # If dice is very high (close to 1), loss will be very small (close to 0)
    # This is normal, but we should check if it's too small
    return total_loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation.

    Args:
        smooth (float): Smoothing factor. Default: 1.
        exponent (float): Exponent factor. Default: 2.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Default: 'mean'.
        class_weight (list[float], optional): Weight of each class.
            Default: None.
        loss_weight (float, optional): Weight of the loss. Default: 1.0.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 use_sigmoid=None,  # Accept but ignore (for compatibility)
                 use_mask=None,      # Accept but ignore (for compatibility)
                 **kwargs):          # Accept any other kwargs for compatibility
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C, H, W) or (N, C, D, H, W).
            target (torch.Tensor): The ground truth with shape (N, H, W) or (N, D, H, W).
            weight (torch.Tensor, optional): Element-wise weights. Default: None.
            avg_factor (int, optional): Average factor. Default: None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method. Default: None.

        Returns:
            torch.Tensor: The calculated Dice loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        
        # Get ignore_index from kwargs if provided
        ignore_idx = kwargs.get('ignore_index', self.ignore_index)
        
        # Create valid mask (ignore ignore_index)
        if ignore_idx is not None:
            valid_mask = (target != ignore_idx).float()
        else:
            valid_mask = None
        
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            valid_mask=valid_mask,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=ignore_idx,
            batch_dice=False)  # Per-sample Dice
        
        return loss


@LOSSES.register_module()
class DiceCELoss(nn.Module):
    """Combined Dice and CrossEntropy Loss for semantic segmentation.
    
    This is similar to SegMamba's DC_and_CE_loss.

    Args:
        dice_weight (float): Weight for Dice loss. Default: 1.0.
        ce_weight (float): Weight for CrossEntropy loss. Default: 1.0.
        smooth (float): Smoothing factor for Dice loss. Default: 1.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Default: 'mean'.
        class_weight (list[float], optional): Weight of each class for CE loss.
            Default: None.
        loss_weight (float, optional): Overall weight of the loss. Default: 1.0.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 dice_weight=1.0,
                 ce_weight=1.0,
                 smooth=1,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 use_sigmoid=None,  # Accept but ignore (for compatibility)
                 use_mask=None,      # Accept but ignore (for compatibility)
                 **kwargs):          # Accept any other kwargs for compatibility
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        
        # Initialize Dice Loss
        self.dice_loss = DiceLoss(
            smooth=smooth,
            reduction=reduction,
            class_weight=None,  # Dice loss doesn't use class_weight
            loss_weight=1.0,
            ignore_index=ignore_index)
        
        # Store class_weight for later use (will be converted to tensor on device)
        self.class_weight = class_weight
        self.reduction = reduction
        
        # Initialize CrossEntropy Loss (weight will be set in forward)
        self.ce_loss = None  # Will be initialized in forward with correct device

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C, H, W) or (N, C, D, H, W).
            target (torch.Tensor): The ground truth with shape (N, H, W) or (N, D, H, W).
            weight (torch.Tensor, optional): Element-wise weights. Default: None.
            avg_factor (int, optional): Average factor. Default: None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method. Default: None.
            ignore_index (int, optional): The label index to be ignored. 
                If provided, will override self.ignore_index. Default: None.

        Returns:
            torch.Tensor: The calculated combined loss.
        """
        # Use provided ignore_index or fall back to self.ignore_index
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index
        
        # Calculate Dice loss
        dice_loss_val = self.dice_loss(
            pred, target, weight, avg_factor, reduction_override, 
            ignore_index=ignore_idx, **kwargs)
        
        # Calculate CrossEntropy loss
        # CE loss expects target to be (N, H, W) with class indices
        # Convert class_weight to tensor on the same device as pred
        ce_weight = None
        if self.class_weight is not None:
            ce_weight = pred.new_tensor(self.class_weight)
        
        # Create CE loss with correct device and ignore_index
        ce_loss = nn.CrossEntropyLoss(
            weight=ce_weight,
            ignore_index=ignore_idx,
            reduction=self.reduction)
        
        ce_loss_val = ce_loss(pred, target.long())
        
        # Combine losses
        combined_loss = self.dice_weight * dice_loss_val + self.ce_weight * ce_loss_val
        
        return self.loss_weight * combined_loss

