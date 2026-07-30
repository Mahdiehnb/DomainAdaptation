import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from SourceModel import MobileNetV2Source
from exp_utils import save_ckpt, load_ckpt

# Training for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels, idx in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)
        #labels = torch.as_tensor([label_map[l.item()] for l in labels], device=device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation / Test
def evaluate(model, dataloader, criterion, device, phase="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels, idx in tqdm(dataloader, desc=phase, leave=False):
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            #labels = torch.as_tensor([label_map[l.item()] for l in labels], device=device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    #print(f'ground truth labels: {labels}, predicted labels: {preds}')
    return epoch_loss, epoch_acc

# Main training loop
def train_source_model(cfg_s, datasets, device="cuda"):
    """
    Train the source model on given datasets.

    Args:
        cfg_s: config object with hyperparams (epochs, lr, etc.)
        datasets: dict with 'train', 'val', 'test' DataLoaders
    Returns:
        exp_dir: Path to experiment directory
        best_val: best validation accuracy
        best_test: test accuracy of best model
    """
    source_num_classes = cfg_s.source_num_classes
    model = MobileNetV2Source(source_num_classes=source_num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg_s.lr, weight_decay=cfg_s.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Lists to save metrics per epoch
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Experiment directory
    exp_dir = Path("runs") / cfg_s.exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    best_test = 0.0

    for epoch in range(cfg_s.epochs):
        print(f"\nEpoch {epoch+1}/{cfg_s.epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(model, datasets["train"], optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = evaluate(model, datasets["val"], criterion, device, phase="Val")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save last checkpoint
        save_ckpt(
            exp_dir / "checkpoints" / "last.pth",
            epoch + 1,
            model,
            optimizer,
            {"best_val": best_val},
        )

        # Save best checkpoint
        if val_acc > best_val:
            best_val = val_acc
            save_ckpt(
                exp_dir / "checkpoints" / "best.pth",
                epoch + 1,
                model,
                optimizer,
                {"best_val": best_val},
            )
            print(f"Saved new best model (Val Acc: {best_val:.4f})")

        # Save the full training history
        history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs
        }
        history_path = exp_dir / "checkpoints" / "training_history.pth"
        torch.save(history, history_path)
        print(f"Training history saved at: {history_path}")

    # Evaluate best checkpoint on test set
    model = MobileNetV2Source(source_num_classes=source_num_classes).to(device)
    ckpt_path = exp_dir / "checkpoints" / "best.pth"
    _, info = load_ckpt(ckpt_path, model, optimizer=None, map_location=device)

    test_loss, test_acc = evaluate(model, datasets["test"], criterion, device, phase="Test")
    best_test = test_acc
    print(f"Best model test accuracy: {test_acc:.4f}")

    return exp_dir, best_val, best_test

# Load frozen source model for SF-OSDA
#def load_frozen_source_model(exp_dir, source_num_classes, device="cuda"):
    #"""
    #Load the best checkpoint of the trained source model in frozen mode.

    #Args:
        #exp_dir (str or Path): Path to the experiment directory.
        #source_num_classes (int): Number of classes the source model was trained with.
        #device (str): Device to load model onto.

    #Returns:
        #model (nn.Module): The source model loaded with best weights, frozen and in eval mode.
    #"""
    #model = MobileNetV2Source(source_num_classes=source_num_classes).to(device)

    #ckpt_path = Path(exp_dir) / "checkpoints" / "best.pth"
    #if not ckpt_path.exists():
        #raise FileNotFoundError(f"No best.pth found in {ckpt_path.parent}")

    #_, info = load_ckpt(ckpt_path, model, optimizer=None, map_location=device)
    #print(f"Loaded source model from {ckpt_path} with best val acc: {info.get('best_val')}")

    #for p in model.parameters():
        #p.requires_grad = False
    #model.eval()

    #return model

def load_source_model(exp_dir, known_classes = Tuple[int], device="cuda"):
    source_num_classes = len(known_classes)
    model = MobileNetV2Source(source_num_classes=source_num_classes).to(device)
    ckpt_path = Path(exp_dir) / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best.pth found in {ckpt_path.parent}")
    _, info = load_ckpt(ckpt_path, model, optimizer=None, map_location=device)
    print(f"Loaded source model from {ckpt_path} with best val acc: {info.get('best_val')}")
    return model
