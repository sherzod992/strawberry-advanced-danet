#!/usr/bin/env python3
"""
Training and evaluation for AG-Transfer ConvNeXt and Table 2 baselines (Section 4.1).

Dataset: strawberry_open_v13 — 4,472 images, 14 classes (train/val/test).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ag_convnext import create_benchmark_model  # noqa: E402
from scripts.quantize import (  # noqa: E402
    dequantize_int8_packed_to_model,
    pack_static_int8_weights,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    classes = train_ds.classes
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Split sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    targets = [y for _, y in train_ds.samples]
    counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: list[str],
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    preds_list: list[np.ndarray] = []
    tgts_list: list[np.ndarray] = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        running_loss += criterion(logits, y).item() * x.size(0)
        preds_list.append(logits.argmax(dim=1).cpu().numpy())
        tgts_list.append(y.cpu().numpy())

    y_true = np.concatenate(tgts_list)
    y_pred = np.concatenate(preds_list)
    n = max(len(y_true), 1)
    labels = list(range(len(classes)))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "loss": float(running_loss / n),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_precision": per_class[0].tolist(),
        "per_class_recall": per_class[1].tolist(),
        "per_class_f1": per_class[2].tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, zero_division=0, output_dict=True
        ),
    }


def save_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train / evaluate AG-Transfer ConvNeXt (14 classes)")
    p.add_argument(
        "--model",
        default="ag_convnext",
        help="ag_convnext | densenet121 | vgg16 | vgg19 | resnet50 | inceptionv3 | "
        "inceptionresnetv2 | mobilenet | nasnetmobile | sknet",
    )
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-dir", default="data/processed/strawberry")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--quantize-int8", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_benchmark_model(
        model_name=args.model,
        num_classes=len(classes),
        pretrained=args.pretrained,
        dropout_p=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    best_fp32_path = results_dir / f"best_{args.model}_fp32_seed{args.seed}.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device, classes)
        scheduler.step(val_m["loss"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_m["accuracy"])

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc * 100:.2f}% | "
            f"val loss {val_m['loss']:.4f} acc {val_m['accuracy'] * 100:.2f}%"
        )

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save({"model": model.state_dict(), "classes": classes}, best_fp32_path)

    ckpt = torch.load(best_fp32_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    fp32_test = evaluate(model, test_loader, criterion, device, classes)
    print(
        f"\n[FP32 test] top-1 {fp32_test['accuracy'] * 100:.2f}% | "
        f"macro-F1 {fp32_test['f1_macro'] * 100:.2f}%"
    )

    save_confusion_matrix(
        np.array(fp32_test["confusion_matrix"]),
        classes,
        figures_dir / f"cm_{args.model}_fp32_seed{args.seed}.png",
        title=f"14-class confusion matrix — {args.model} (FP32)",
    )

    int8_metrics = None
    if args.quantize_int8 and args.model.lower() in (
        "ag_convnext",
        "ag_transfer_convnext",
        "proposed",
    ):
        int8_path = results_dir / f"best_{args.model}_int8_packed_seed{args.seed}.pt"
        packed_state, size_int8_mb = pack_static_int8_weights(model, int8_path)
        q_model = dequantize_int8_packed_to_model(model, packed_state).to(device)
        int8_metrics = evaluate(q_model, test_loader, criterion, device, classes)
        int8_metrics["size_mb"] = round(size_int8_mb, 2)
        print(
            f"[INT8 packed] top-1 {int8_metrics['accuracy'] * 100:.2f}% | "
            f"macro-F1 {int8_metrics['f1_macro'] * 100:.2f}% | "
            f"size {size_int8_mb:.2f} MB"
        )

    out_json = results_dir / f"result_{args.model}_seed{args.seed}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "seed": args.seed,
                "epochs": args.epochs,
                "best_val_acc": best_val_acc,
                "fp32_test_metrics": fp32_test,
                "int8_test_metrics": int8_metrics,
                "classes": classes,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved report: {out_json}")


if __name__ == "__main__":
    main()
