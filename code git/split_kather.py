import os
import os.path as osp
import random

random.seed(42)  # reproducible train/val/test splits


def safe_makedir(path):
    os.makedirs(path, exist_ok=True)


def link_unique(src, dst_dir):
    """
    Create a symlink for src inside dst_dir instead of physically copying it.
    This makes the split step near-instant and uses zero extra disk space —
    Kaggle input datasets are read-only and don't need to be duplicated on
    disk just to be organized into train/val/test folders.
    """
    dst = osp.join(dst_dir, osp.basename(src))
    if not osp.lexists(dst):  # lexists also catches broken symlinks
        os.symlink(src, dst)


def _split_dataset(path_data_in, base_out, train_frac, val_frac, test_frac):
    path_train = osp.join(base_out, "train")
    path_val = osp.join(base_out, "val")
    path_test = osp.join(base_out, "test")

    for p in [path_train, path_val, path_test]:
        safe_makedir(p)

    if any(os.listdir(p) for p in [path_train, path_val, path_test]):
        print(f"Train/val/test folders already contain files in {base_out}. Skipping split.")
        return

    subfolders = [
        f for f in os.listdir(path_data_in)
        if osp.isdir(osp.join(path_data_in, f))
    ]

    for f in subfolders:
        for p in [path_train, path_val, path_test]:
            safe_makedir(osp.join(p, f))

        src_dir = osp.join(path_data_in, f)
        im_list = os.listdir(src_dir)
        n = len(im_list)

        test_len = int(test_frac * n)
        val_len = int(val_frac * n)

        im_test = random.sample(im_list, test_len)
        remaining = list(set(im_list) - set(im_test))

        im_val = random.sample(remaining, val_len)
        im_train = list(set(remaining) - set(im_val))

        for im in im_train:
            link_unique(osp.join(src_dir, im), osp.join(path_train, f))
        for im in im_val:
            link_unique(osp.join(src_dir, im), osp.join(path_val, f))
        for im in im_test:
            link_unique(osp.join(src_dir, im), osp.join(path_test, f))

    print(f"Split complete (symlinked, no data duplicated).\nSaved to: {base_out}")


def split_kather16(path_data_in):
    base_out = "/kaggle/working/kather16"
    _split_dataset(path_data_in, base_out, train_frac=0.70, val_frac=0.15, test_frac=0.15)


def split_kather19(path_data_in):
    base_out = "/kaggle/working/kather19"
    _split_dataset(path_data_in, base_out, train_frac=0.70, val_frac=0.15, test_frac=0.15)