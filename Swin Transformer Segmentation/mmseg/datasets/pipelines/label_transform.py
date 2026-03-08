import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class ConvertLabels(object):
    """Convert label values to target values.
    
    This is useful when labels are stored as 0 and 255 (common in medical images)
    but need to be converted to 0 and 1 for training.
    
    Reference: SegMamba loads labels directly from .npy files without conversion,
    but for ISIC2017, labels are stored as PNG with values 0 and 255.
    
    Args:
        label_mapping (dict): Mapping from old label values to new label values.
            Example: {0: 0, 255: 1} to convert 255 to 1.
        ignore_index (int): Label value to ignore. Default: 255.
    """
    
    def __init__(self, label_mapping=None, ignore_index=255):
        self.label_mapping = label_mapping
        self.ignore_index = ignore_index
    
    def __call__(self, results):
        """Call function to convert labels.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Result dict with converted labels.
        """
        if 'gt_semantic_seg' not in results:
            return results
        
        gt_semantic_seg = results['gt_semantic_seg']
        
        # Create a copy to avoid modifying the original
        converted_seg = gt_semantic_seg.copy()
        
        if self.label_mapping is not None:
            # Apply label mapping
            # Process in order to avoid conflicts (e.g., if mapping 255->1 and 1->2)
            # Sort by old_id in descending order to process larger values first
            sorted_mapping = sorted(self.label_mapping.items(), key=lambda x: x[0], reverse=True)
            for old_id, new_id in sorted_mapping:
                converted_seg[gt_semantic_seg == old_id] = new_id
        
        # Convert any remaining non-mapped values to ignore_index
        # This handles cases where labels have unexpected values
        if self.label_mapping is not None:
            # Get all valid label values from mapping
            valid_values = set(self.label_mapping.values())
            # Set invalid values to ignore_index
            invalid_mask = ~np.isin(converted_seg, list(valid_values) + [self.ignore_index])
            if invalid_mask.any():
                converted_seg[invalid_mask] = self.ignore_index
        
        results['gt_semantic_seg'] = converted_seg
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(label_mapping={self.label_mapping},'
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str

