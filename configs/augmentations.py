import random
from albumentations.core.transforms_interface import BasicTransform
from torch.nn import functional as F
from albumentations import Compose, random_utils
import torch
import numpy as np
import math
import typing

# https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/blob/main/configs/augmentations.py
class TemporalMask(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        mask_value=0,
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalMask, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], **params):
        l = data[:,1:-1].shape[1]
        x_new = data.clone()
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,1,l)
            mask_offset = int(mask_offset_01 * max_mask)
            x_new[:,mask_offset+1:mask_offset+mask_size-1] = torch.tensor(self.mask_value)
        return x_new

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                'mask_value':self.mask_value,}

    def get_transform_init_args_names(self):
        return ("size","mask_value","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  

class TemporalMask2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        mask_value=0,
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalMask2, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], **params):
        l = data.shape[1]
        x_new = data.clone()
        # print(mask_sizes,mask_offsets_01)
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,0,l)
            mask_offset = int(mask_offset_01 * max_mask)
            # print(l, mask_offset,mask_offset+mask_size-1)
            x_new[:,mask_offset:mask_offset+mask_size-1] = self.mask_value
        return x_new

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                'mask_value':self.mask_value,}

    def get_transform_init_args_names(self):
        return ("size","mask_value","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  
