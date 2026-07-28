""" This script is inspired from https://github.com/LTS5/Distill-SODA/blob/main/datasets/utils.py """

import numpy as np
import torch
import torchvision
from typing import List
from PIL import Image
from torch.utils.data import ConcatDataset

# Opens an image file and makes sure it's RGB.
def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            my_img = img.convert('RGB')
        return my_img

# Extends PyTorch's ImageFolder.
# ImageFolder returns (img, label).
# This version can also return the index of the sample (if return_idx=True).
# It is useful for tracking individual samples later (for Open-Set Recognition).

class IndexedImageFolder(torchvision.datasets.ImageFolder):
    """ ImageFolder with indexes
    """

    def __init__(self, root : str, transform, return_idx : bool):
        """Initialize the dataset
        Args:
            root (str): path to the dataset
            transform (torchvision.transforms): transform function to apply to images
            return_idx (bool): whether the dataset return index when an image is requested.
        """
        super().__init__(root, transform, loader=rgb_loader) 
        self.return_idx = return_idx

    def __getitem__(self, item):
        img, label = super().__getitem__(item)

        if self.return_idx:
            return img, label, item # item = index of the sample in the dataset
        return img, label

    def collate_fn_with_idx(batch):
        imgs, labels, idxs = zip(*batch)
        imgs = torch.stack(imgs, dim=0) # shape: (B, C, H, W)
        labels = torch.tensor(labels, dtype=torch.long)
        idxs = torch.tensor(idxs, dtype=torch.long)
        return imgs, labels, idxs

# Ensures that if return_idx=True, you still get back a global index across the concatenated dataset;
# so you can trace back which original dataset it came from.

class IndexedConcatDataset(ConcatDataset):
    """ ConcatDataset Extension for IndexedImageFolder objects
    """

    def __init__(self, datasets : List[IndexedImageFolder]) -> None:
        """ Initialize the dataste with a list of IndexedImageFolder to concatenate.

        Args:
            datasets (List[IndexedImageFolder]): a list of IndexedImageFolder.
        """
        super().__init__(datasets)
        self.return_idx = datasets[0].return_idx

    def __getitem__(self, idx):
        if self.return_idx:
            img, label, _ = super().__getitem__(idx)
            return img, label, idx
        else:
            return super().__getitem__(idx)

# Filters a dataset to only keep the samples at positions idxs
def subsample_dataset(dataset, idxs):
    imgs, sampls = [], []
    for i in idxs:
        imgs.append(dataset.imgs[i])
        sampls.append(dataset.samples[i])
    dataset.imgs = imgs
    dataset.samples = sampls
    dataset.targets = np.array(dataset.targets)[idxs].tolist()

    return dataset

# ***** from the distill-soda paper code *****
# # Filters a dataset to only specific classe
# def subsample_classes(dataset, include_classes=(0,1), is_ood=False):
#     cls_idxs = []

#     target_xform_dict = {}
#     i = -1
#     for k in include_classes:

#         if not is_ood:
#             i += 1

#         if isinstance(k, tuple):
#             for l in k:
#                 target_xform_dict[l] = i
#                 cls_idxs += [x for x,y in enumerate(dataset.targets) if y == l]
#         else:
#             target_xform_dict[k] = i
#             cls_idxs += [x for x,y in enumerate(dataset.targets) if y == k]
     
#     dataset = subsample_dataset(dataset, cls_idxs)
#     dataset.target_transform = lambda x: target_xform_dict[x]

#     new_class_to_idx = {key: val for key, val in dataset.class_to_idx.items() if val in target_xform_dict.keys()}
#     new_class_to_idx = {k: target_xform_dict[v] for k, v in new_class_to_idx.items()}
#     dataset.class_to_idx = new_class_to_idx
#     dataset.idx_to_class = dict((v, k) for k, v in new_class_to_idx.items())

#     return dataset

# ***** we wrote this to make the classes start from 0 in source and target so we can map them, 
# but unknown classes in the target test set are being remapped again from 0. *****
# def subsample_classes(dataset, include_classes, is_ood=False):
#     merged_mapping = {}
#     class_counter = 0
#     for cls in include_classes:
#         if isinstance(cls, tuple):
#             for subcls in cls:
#                 merged_mapping[subcls] = class_counter
#         else:
#             merged_mapping[cls] = class_counter
#         class_counter += 1

#     # Remap dataset.targets
#     new_targets = []
#     new_samples = []
#     for i, (path, label) in enumerate(dataset.samples):
#         if label in merged_mapping:
#             new_targets.append(merged_mapping[label])
#             new_samples.append((path, merged_mapping[label]))
#     dataset.samples = new_samples
#     dataset.targets = new_targets
#     dataset.classes = list(range(len(include_classes)))
#     dataset.is_ood = is_ood

#     return dataset

def subsample_classes(dataset, include_classes, is_ood=False, offset=None):
    # If offset is None:
    # start at 0 for known classes
    # start at len(include_classes) for unknown classes
    if offset is None:
        offset = 0 if not is_ood else len(include_classes)

    merged_mapping = {}
    class_counter = offset
    for cls in include_classes:
        if isinstance(cls, tuple):
            for subcls in cls:
                merged_mapping[subcls] = class_counter
        else:
            merged_mapping[cls] = class_counter
        class_counter += 1

    # Remap dataset.targets
    new_targets = []
    new_samples = []
    for i, (path, label) in enumerate(dataset.samples):
        if label in merged_mapping:
            new_targets.append(merged_mapping[label])
            new_samples.append((path, merged_mapping[label]))
    # dataset.samples = new_samples
    # dataset.targets = new_targets
    # dataset.classes = list(range(len(set(new_targets))))
    # dataset.is_ood = is_ood

    # return dataset
    dataset.samples = new_samples
    dataset.targets = new_targets

    # Derive classes from the INTENDED mapping (include_classes), not from
    # whatever labels happen to survive subsampling — an empty class must
    # still occupy its slot, or downstream indices silently misalign.
    expected_classes = sorted(set(merged_mapping.values()))
    dataset.classes = expected_classes

    present_classes = set(new_targets)
    missing = set(expected_classes) - present_classes
    if missing:
        print(f"[subsample_classes] Warning: no samples found for "
              f"target class index/indices {sorted(missing)} "
              f"(is_ood={is_ood}). dataset.classes still includes them.")

    dataset.is_ood = is_ood

    return dataset
