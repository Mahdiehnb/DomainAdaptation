""" This script is inspired from https://github.com/LTS5/Distill-SODA/blob/main/datasets/histo.py
    Adapted for Kather16 and Kather19 datasets.
"""

import torch
import torchvision
import os
import os.path as osp
import torch.multiprocessing
import torchvision.transforms as transforms
from typing import Tuple

from utils_data import subsample_classes, IndexedConcatDataset, IndexedImageFolder

torch.multiprocessing.set_sharing_strategy('file_system')

# ******* Dataset splits  (for Kather16 & Kather19) *******
osr_splits = {
    # *** KATHER 16 ***
    # Classes: ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS','06_MUCOSA', '07_ADIPOSE', '08_EMPTY']
    'kather16': {
        "n_classes": 8,
        "splits": [
            [0, 1, 3, 5], # S1: TUMOR, STROMA, LYMPHO, MUCOSA
            [0, 1], # S2: TUMOR, STROMA
            [0, 1, 3], # S3: TUMOR, STROMA, LYMPHO
        ],
    },

    # *** KATHER 19 ***
    # Classes: ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    'kather19': {
        "n_classes": 9,
        "splits": [
            [8, (5, 7), 3, 6], # S1: TUM, (MUS,STR), LYM, NORM → TUMOR, STROMA, LYMPHO, MUCOSA
            [8, (5, 7)], # S2: TUM, (MUS,STR) → TUMOR, STROMA
            [8, (5, 7), 3], # S3: TUM, (MUS,STR), LYM → TUMOR, STROMA, LYMPHO
        ],
    },
}

# ******* Transforms ******* 
# We use the MobileNetV2 network.
# MobileNetV2 uses input images of size 224×224.
def get_histopathology_transform(image_size=224):
    """Generate train/test transforms for histopathology images."""

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, test_transform

# ******* Data Loader: Source Dataset *******
def load_source_dataset(root_dir: str,
                         train_transform,
                         test_transform,
                         known_classes: Tuple[int],
                         unknown_classes: Tuple[int],
                         return_idx: bool = False):
    """Load source dataset splits for training, validation, and testing."""

    train_dir = osp.join(root_dir, 'train')
    val_dir = osp.join(root_dir, 'val')
    test_dir = osp.join(root_dir, 'test')

    # Train
    train_dataset = IndexedImageFolder(root=train_dir, transform=train_transform, return_idx=return_idx)
    train_dataset = subsample_classes(train_dataset, include_classes=known_classes)
    print("Training Known Dataset size:", len(train_dataset))

    # Validation
    val_known = IndexedImageFolder(root=val_dir, transform=test_transform, return_idx=return_idx)
    val_known = subsample_classes(val_known, include_classes=known_classes)

    val_unknown = IndexedImageFolder(root=val_dir, transform=test_transform, return_idx=return_idx)
    val_unknown = subsample_classes(val_unknown, include_classes=unknown_classes, is_ood=True)

    # Test
    test_known = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_known = subsample_classes(test_known, include_classes=known_classes)

    test_unknown = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_unknown = subsample_classes(test_unknown, include_classes=unknown_classes, is_ood=True)

    all_datasets = {
        'train': train_dataset,
        'val_known': val_known,
        'val_unknown': val_unknown,
        'val': IndexedConcatDataset([val_known, val_unknown]),
        'test_known': test_known,
        'test_unknown': test_unknown,
        'test': IndexedConcatDataset([test_known, test_unknown]),
    }

    return all_datasets

# ******* Data Loader: Target Dataset *******
def load_target_dataset(root_dir: str,
                         train_transform,
                         test_transform,
                         known_classes: Tuple[int],
                         unknown_classes: Tuple[int],
                         return_idx: bool = False):
    """Load target dataset splits (unlabeled train, known/unknown test)."""

    train_dir = osp.join(root_dir, 'train')
    test_dir = osp.join(root_dir, 'test')

    # Train
    train_known = IndexedImageFolder(root=train_dir, transform=test_transform, return_idx=return_idx) 
    train_known = subsample_classes(train_known, include_classes=known_classes)

    train_unknown = IndexedImageFolder(root=train_dir, transform=test_transform, return_idx=return_idx)
    train_unknown = subsample_classes(train_unknown, include_classes=unknown_classes, is_ood=True, offset=len(known_classes))

    # Test
    test_known = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_known = subsample_classes(test_known, include_classes=known_classes)

    test_unknown = IndexedImageFolder(root=test_dir, transform=test_transform, return_idx=return_idx)
    test_unknown = subsample_classes(test_unknown, include_classes=unknown_classes, is_ood=True, offset=len(known_classes))

    all_datasets = {
        'train': IndexedConcatDataset([train_known, train_unknown]),
        'test_known': test_known,
        'test_unknown': test_unknown,
        'test': IndexedConcatDataset([test_known, test_unknown]),
    }

    return all_datasets

# Main entry
def get_histopathology_datasets(root_dir: str,
                                name: str,
                                split_idx: int,
                                transform=None,
                                image_size: int = 224,
                                target: bool = True,
                                return_idx: bool = True):
    """ Main entry point: load histopathology dataset. """

    assert name in ['kather16', 'kather19'], "kather16 and kather19 datasets supported."
    print(f"\nLoading dataset {name}")

    dataset_info = osr_splits[name]
    root_dir = osp.join(root_dir, name) 

    max_n_classes = dataset_info["n_classes"]
    known_classes = dataset_info["splits"][split_idx - 1]
    n_classes = len(known_classes)

    # flatten tuples
    known_classes_f = []
    for c in known_classes:
        if isinstance(c, tuple):
            known_classes_f += list(c)
        else:
            known_classes_f.append(c)

    unknown_classes = [x for x in range(max_n_classes) if x not in known_classes_f]
    print(f'{name} known classes: {known_classes}')
    print(f'{name} open set classes: {unknown_classes}')

    # transforms
    if transform is not None:
        train_transform = test_transform = transform
    else:
        train_transform, test_transform = get_histopathology_transform(image_size=image_size)

    # Load dataset splits
    if target:
        datasets = load_target_dataset(root_dir, train_transform, test_transform, known_classes, 
                                            unknown_classes, return_idx=return_idx)
    else:
        datasets = load_source_dataset(root_dir, train_transform, test_transform,
                                            known_classes, unknown_classes, return_idx=return_idx)

    return datasets, n_classes

