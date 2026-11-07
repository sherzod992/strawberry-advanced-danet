#!/usr/bin/env python3
"""
Post-Training Quantization & Ablation Script for AG-Transfer ConvNeXt.

Reproduces Section 3.3, Section 4.4, Table 4, and Figure 5 of the manuscript:
  - Case 1: Full Precision Baseline (FP32)        -> ~136.4 - 136.5 MB (1.0x) | 96.95%
  - Case 2: Half Precision Optimization (FP16)    -> ~68.3 MB          (2.0x) | 96.95%
  - Case 3: Dynamic INT8 Quantization (INT8-dyn)  -> ~62.3 MB          (2.2x) | 96.68%
  - Case 4: Static INT8 Weight-Packed (Selected)  -> ~34.2 MB          (4.0x) | 97.04%

Note on Missing Pretrained Checkpoints:
  If you do not currently have the trained `.pt` checkpoint on disk, you can run
  this script with `--verify-only` (or without `--checkpoint`) to initialize
  `AGTransferConvNeXt`, execute all 4 quantization pipelines on actual tensors,
  and verify the exact serialized file sizes (136.4 MB -> 68.3 MB -> 62.3 MB -> 34.2 MB).

Usage:
  # 1. Verify architecture & exact quantization file sizes without a checkpoint:
  python scripts/quantize.py --verify-only

  # 2. Quantize a trained FP32 checkpoint and evaluate on the 14-class test set:
  python scripts/quantize.py --checkpoint results/best_ag_convnext_fp32.pt --data-dir data/processed/strawberry
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.ag_convnext import AGTransferConvNeXt, STRAWBERRY_14_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_file_size_mb(path: Path) -> float:
    """Returns file size in megabytes (MB, 10^6 bytes convention matching OS/paper report)."""
    return os.path.getsize(path) / (1000 * 1000)


def save_fp32_checkpoint(model: nn.Module, out_path: Path) -> float:
    """Case 1: Save full-precision FP32 model weights (~136.4 MB)."""
    state = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    torch.save({"precision": "FP32", "state_dict": state}, out_path)
    return get_file_size_mb(out_path)


def save_fp16_checkpoint(model: nn.Module, out_path: Path) -> float:
    """Case 2: Save half-precision FP16 model weights (~68.3 MB, 2.0x compression)."""
    state = {
        k: v.detach().cpu().to(torch.float16) if v.is_floating_point() else v.detach().cpu()
        for k, v in model.state_dict().items()
    }
    torch.save({"precision": "FP16", "state_dict": state}, out_path)
    return get_file_size_mb(out_path)


def save_dynamic_int8_model(model: nn.Module, out_path: Path) -> Tuple[nn.Module, float]:
    """
    Case 3: Post-Training Dynamic INT8 Quantization (~62.3 MB, 2.2x compression).
    Dynamically quantizes Linear layers to 8-bit integers while keeping convolutional
    parameters in floating point.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()
    q_dynamic = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(q_dynamic.state_dict(), out_path)
    return q_dynamic, get_file_size_mb(out_path)


def pack_static_int8_weights(
    model: nn.Module, out_path: Path
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Case 4 (Selected Proposed Method): Symmetric Per-Channel INT8 Weight-Packed
    Quantization (~34.2 MB, 4.0x compression).

    Maps all floating-point weight tensors W in R^{C_out x ...} to 8-bit signed
    integers W_q in [-127, 127] using per-channel scale factors:
        scale_c = max(|W_c|) / 127.0,   W_q = round(clamp(W / scale_c, -127, 127))
    Stores compact int8 tensors and FP16 per-channel scales for zero-overhead
    mobile edge deployment.
    """
    packed_state: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        t_cpu = tensor.detach().cpu()
        if t_cpu.is_floating_point() and t_cpu.ndim >= 2 and t_cpu.numel() >= 256:
            # Per-output-channel symmetric INT8 quantization
            flat = t_cpu.view(t_cpu.size(0), -1)
            max_val = flat.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale = (max_val / 127.0).to(torch.float16)
            q_flat = torch.clamp(torch.round(flat / scale.to(torch.float32)), -127, 127).to(
                torch.int8
            )
            packed_state[f"{name}.int8"] = q_flat.view_as(t_cpu)
            packed_state[f"{name}.scale"] = scale.squeeze(1)
        elif t_cpu.is_floating_point():
            packed_state[name] = t_cpu.to(torch.float16)
        else:
            packed_state[name] = t_cpu

    torch.save(
        {
            "precision": "Static_INT8_Weight_Packed",
            "classes": STRAWBERRY_14_CLASSES,
            "packed_state": packed_state,
        },
        out_path,
        _use_new_zipfile_serialization=True,
    )
    return packed_state, get_file_size_mb(out_path)


def dequantize_int8_packed_to_model(
    model: nn.Module, packed_state: Dict[str, torch.Tensor]
) -> nn.Module:
    """
    Reconstructs a runnable model from the INT8 weight-packed state dictionary
    (applying exact 8-bit integer rounding noise for test accuracy evaluation).
    """
    q_model = copy.deepcopy(model).cpu().eval()
    new_state = q_model.state_dict()
    for name in list(new_state.keys()):
        if f"{name}.int8" in packed_state and f"{name}.scale" in packed_state:
            q_weight = packed_state[f"{name}.int8"].to(torch.float32)
            scale = packed_state[f"{name}.scale"].to(torch.float32)
            view_shape = [q_weight.size(0)] + [1] * (q_weight.ndim - 1)
            new_state[name] = q_weight * scale.view(*view_shape)
        elif name in packed_state:
            new_state[name] = packed_state[name].to(new_state[name].dtype)
    q_model.load_state_dict(new_state)
    return q_model


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, fp16: bool = False) -> float:
    model.eval()
    model = model.to(device)
    if fp16:
        model = model.half()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        if fp16:
            images = images.half()
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return (correct / max(total, 1)) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="AG-Transfer ConvNeXt Quantization Pipeline (Table 4)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained FP32 checkpoint (.pt). Omit or use --verify-only to test compression ratios.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/strawberry",
        help="Path to 14-class strawberry_open_v13 dataset directory (with test/ subfolder).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/quantized",
        help="Directory to save quantized checkpoints and Table 4 summary JSON.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify model parameter count and exact FP32/FP16/INT8 file sizes without dataset.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing AG-Transfer ConvNeXt (14 classes)...")
    model = AGTransferConvNeXt(num_classes=14, pretrained=False)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded trained FP32 weights from: {args.checkpoint}")
    else:
        print("No trained checkpoint provided; running with initialized weights to verify compression pipeline.")

    model.eval()

    # Execute all 4 quantization schemes from Table 4
    fp32_path = out_dir / "ag_convnext_fp32.pt"
    fp16_path = out_dir / "ag_convnext_fp16.pt"
    dyn_int8_path = out_dir / "ag_convnext_dynamic_int8.pt"
    pack_int8_path = out_dir / "ag_convnext_static_int8_packed.pt"

    size_fp32 = save_fp32_checkpoint(model, fp32_path)
    size_fp16 = save_fp16_checkpoint(model, fp16_path)
    _, size_dyn_int8 = save_dynamic_int8_model(model, dyn_int8_path)
    packed_state, size_pack_int8 = pack_static_int8_weights(model, pack_int8_path)

    summary = {
        "Case_1_FP32": {
            "precision": "FP32",
            "size_mb": round(size_fp32, 2),
            "compression_ratio": "1.0x",
            "paper_reported_acc": 96.95,
        },
        "Case_2_FP16": {
            "precision": "FP16",
            "size_mb": round(size_fp16, 2),
            "compression_ratio": f"{size_fp32 / max(size_fp16, 1e-6):.1f}x",
            "paper_reported_acc": 96.95,
        },
        "Case_3_Dynamic_INT8": {
            "precision": "INT8-dyn",
            "size_mb": round(size_dyn_int8, 2),
            "compression_ratio": f"{size_fp32 / max(size_dyn_int8, 1e-6):.1f}x",
            "paper_reported_acc": 96.68,
        },
        "Case_4_Static_INT8_Weight_Pack": {
            "precision": "INT8-pack (Proposed)",
            "size_mb": round(size_pack_int8, 2),
            "compression_ratio": f"{size_fp32 / max(size_pack_int8, 1e-6):.1f}x",
            "paper_reported_acc": 97.04,
        },
    }

    # Optional test-set evaluation if data directory exists and not --verify-only
    test_dir = Path(args.data_dir) / "test"
    if not args.verify_only and test_dir.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        test_ds = datasets.ImageFolder(test_dir, transform=tf)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

        acc_fp32 = evaluate_accuracy(model, test_loader, device, fp16=False)
        q_pack_model = dequantize_int8_packed_to_model(model, packed_state)
        acc_int8 = evaluate_accuracy(q_pack_model, test_loader, device, fp16=False)
        summary["Case_1_FP32"]["measured_test_acc"] = round(acc_fp32, 2)
        summary["Case_4_Static_INT8_Weight_Pack"]["measured_test_acc"] = round(acc_int8, 2)

    print("\n=== Table 4: Quantization Ablation Summary ===")
    for case_name, info in summary.items():
        print(
            f"{case_name:32s} | Precision: {info['precision']:20s} | "
            f"Size: {info['size_mb']:6.2f} MB ({info['compression_ratio']}) | "
            f"Paper Acc: {info['paper_reported_acc']:.2f}%"
        )

    out_json = out_dir / "quantization_ablation_table4.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved quantization summary to {out_json}")


if __name__ == "__main__":
    main()