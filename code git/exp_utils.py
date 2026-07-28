import torch
import os

def save_ckpt(path, epoch, model, optimizer=None, info: dict = None):
    """
    Save a checkpoint.

    Args:
        path (str or Path): Where to save the checkpoint.
        model (nn.Module): Model to save.
        optimizer (torch.optim.Optimizer, optional): Optimizer state to save.
        info (dict, optional): Extra information to save (e.g., best_val).
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if info is not None:
        state["info"] = info

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    """
    Load a checkpoint.

    Args:
        path (str or Path): Path to checkpoint file.
        model (nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        map_location (str): Device to map checkpoint to.

    Returns:
        (epoch, info_dict)
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    info = ckpt.get("info", {})
    epoch = ckpt.get("epoch", 0)

    return epoch, info
