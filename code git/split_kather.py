import os
import os.path as osp
import random
import shutil

random.seed(42) # Every run of split_kather will give the same train/val/test split.

def safe_makedir(path):
    os.makedirs(path, exist_ok=True)

def copy_unique(src, dst):
    """ Copy a file only if it does not already exist in the destination. """
    if not osp.exists(osp.join(dst, osp.basename(src))):
        shutil.copy2(src, dst)

# ******* Split Kather 16 Dataset into Train/Val/Test *******
def split_kather16(path_data_in):
    base_out = osp.join(osp.dirname(path_data_in), 'kather16')
    path_train = osp.join(base_out, 'train')
    path_val = osp.join(base_out, 'val')
    path_test = osp.join(base_out, 'test')

    for p in [path_train, path_val, path_test]:
        safe_makedir(p)

    # If train/val/test folders already exist, skip
    if any(os.listdir(p) for p in [path_train, path_val, path_test]):
     print("Train/val/test folders already contain files. Skipping split to avoid duplicates.")
     return
     
    subfolders = [f for f in os.listdir(path_data_in) if osp.isdir(osp.join(path_data_in, f))]
    for f in subfolders:
        for p in [path_train, path_val, path_test]:
            safe_makedir(osp.join(p, f))

        im_list = os.listdir(osp.join(path_data_in, f))
        val_len = int(0.15 * len(im_list)) # 15% validation data
        test_len = int(0.15 * len(im_list)) # 15% testing data

        im_test  = random.sample(im_list, test_len)
        remaining = list(set(im_list) - set(im_test))
        im_val = random.sample(remaining, val_len)
        im_train = list(set(remaining) - set(im_val)) # 70% training data

        for im in im_train:
            copy_unique(osp.join(path_data_in, f, im), osp.join(path_train, f))
        for im in im_val:
            copy_unique(osp.join(path_data_in, f, im), osp.join(path_val, f))
        for im in im_test:
            copy_unique(osp.join(path_data_in, f, im), osp.join(path_test, f))

    print("Kather16 split complete (files copied).")

# ******* Split Kather 19 Dataset into Train/Val/Test *******
def split_kather19(path_data_in):
    base_out = osp.join(osp.dirname(path_data_in), 'kather19')
    path_train = osp.join(base_out, 'train')
    path_val = osp.join(base_out, 'val')
    path_test = osp.join(base_out, 'test')

    for p in [path_train, path_val, path_test]:
        safe_makedir(p)

    # If train/val/test folders already exist, skip
    if any(os.listdir(p) for p in [path_train, path_val, path_test]):
     print("Train/val/test folders already contain files. Skipping split to avoid duplicates.")
     return
     
    subfolders = [f for f in os.listdir(path_data_in) if osp.isdir(osp.join(path_data_in, f))]
    for f in subfolders:
        for p in [path_train, path_val, path_test]:
            safe_makedir(osp.join(p, f))

        im_list = os.listdir(osp.join(path_data_in, f))
        train_len = int(0.7 * len(im_list)) # 70% training data
        val_len = int(0.15 * len(im_list)) # 15% validation data
        test_len = int(0.15 * len(im_list)) # 15% testing data

        im_train = random.sample(im_list, train_len)
        remaining = list(set(im_list) - set(im_train))
        im_val = random.sample(remaining, val_len)
        im_test = list(set(remaining) - set(im_val))

        for im in im_train:
            copy_unique(osp.join(path_data_in, f, im), osp.join(path_train, f))
        for im in im_val:
            copy_unique(osp.join(path_data_in, f, im), osp.join(path_val, f))
        for im in im_test:
            copy_unique(osp.join(path_data_in, f, im), osp.join(path_test, f))

    print("Kather19 split complete (files copied).")
 