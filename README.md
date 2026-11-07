# AG-Transfer ConvNeXt: Lightweight Attention-Guided ConvNeXt for Strawberry Pest and Disease Diagnosis

Official PyTorch implementation of "Lightweight Attention-Guided ConvNeXt for Precise Early-Stage Strawberry Pest and Disease Diagnosis on Greenhouse Mobile Edge Devices" (Attention-Guided ConvNeXt / AG-Transfer ConvNeXt).

---

## 1. Overview & Architecture

Deploying deep convolutional neural networks on handheld smartphones in commercial strawberry greenhouses requires resolving both multi-scale symptom variations (from 1 mm thrips/spider mite colonies to large necrotic blights) and strict mobile memory constraints.

Our proposed AG-Transfer ConvNeXt couples an ImageNet-pretrained ConvNeXt-Tiny backbone ($C = 768$) with a Decoupled Selective Kernel (SK) Attention classification head and Post-Training INT8 Weight-Packed Quantization:

- **Heterogeneous Grouped Convolutions (Split):** Parallel $3 \times 3$ and $5 \times 5$ grouped convolutions across cardinal groups ($G = 4$) simultaneously extract micro-scale insect feeding marks and macroscopic foliar lesions.

- **Decoupled SK Attention (No BN + Branch Dropout):**
  - **Decoupled Global Average Pooling:** Instead of element-wise summing multi-scale branches prior to projection, each branch ($U_1$, $U_2$) is pooled independently ($s_1 = \mathrm{GAP}(U_1)$, $s_2 = \mathrm{GAP}(U_2) \in \mathbb{R}^C$) and routed into a shared linear projection ($C \rightarrow C/16$).
  - **Batch Normalization Removal:** Eliminates Batch Normalization (BN) between the attention dense layers so subtle lesion activation boundaries are not artificially compressed.
  - **Independent Branch Dropout ($p = 0.2$):** Applied independently after the shared projection in each branch to prevent feature homogenization and enforce branch orthogonality before channel-wise Softmax fusion ($V = a \cdot U_1 + b \cdot U_2$).

- **Post-Training INT8 Weight-Packed Quantization:** Compresses the full-precision FP32 model (136.4 MB, 96.95% test accuracy) by $4.0\times$ down to 34.2 MB while achieving 97.04% Top-1 test accuracy (93.22% Macro-F1) across 14 classes.

---

## 2. Repository Structure

```text
strawberry-advanced-danet/
├── README.md                 # Documentation, benchmark tables, and Data Availability Statement
├── requirements.txt          # Python package dependencies (PyTorch, timm, scikit-learn, etc.)
├── models/
│   └── ag_convnext.py        # AG-Transfer ConvNeXt & Decoupled SK Attention + 9 Table 2 baselines
├── scripts/
│   ├── prepare_dataset.py    # Build strawberry_open_v13 ImageFolder splits (Section 3)
│   └── quantize.py           # FP32 -> FP16 / Dynamic INT8 / Static Weight-Packed INT8 quantization (Table 4)
├── configs/dataset/
│   ├── strawberry_open_v13_expected_splits.json   # Paper-fixed train/val/test counts per class
│   ├── source_mapping.example.json              # Afzaal / IP102 / PlantVillage merge template
│   └── manifest.example.csv
└── experiments/
    └── train.py              # 14-class training, evaluation, Confusion Matrix, and Grad-CAM pipeline
```

Training outputs (checkpoints, JSON) are written under `results/`; confusion matrices under `figures/`. Raw and processed images live under `data/` (gitignored). Split **counts** are versioned under `configs/dataset/`.

---

## 3. Dataset (strawberry_open_v13 — 14 Classes, 4,472 Images)

The reproducible benchmark dataset (strawberry_open_v13) comprises 4,472 images across 14 classes (6 pests, 7 diseases, 1 healthy), partitioned into 2,818 training (63%), 538 validation (12%), and 1,116 test (25%) images prior to on-the-fly training augmentation:

| Category | Class Folder Name | Description (KR / EN) | Train | Val | Test | Total |
|----------|-------------------|------------------------|------:|----:|-----:|------:|
| Pest | pest_armyworm | 담배거세미나방 (Cotton leafworm) | 141 | 24 | 41 | 206 |
| Pest | pest_thrips | 총채벌레 (Thrips) | 33 | 5 | 9 | 47 |
| Pest | pest_aphid | 진딧물 (Aphids) | 596 | 105 | 175 | 876 |
| Pest | pest_plant_bug | 노린재 (Plant bug) | 237 | 41 | 70 | 348 |
| Pest | pest_spider_mite | 점박이응애 (Two-spotted spider mite) | 109 | 19 | 32 | 160 |
| Pest | pest_whitefly | 온실가루이 (White fly) | 72 | 12 | 21 | 105 |
| Disease | anthracnose_fruit_rot | 탄저병-과실 (Anthracnose fruit rot) | 52 | 12 | 33 | 97 |
| Disease | gray_mold | 잿빛곰팡이병 (Gray mold) | 255 | 77 | 145 | 477 |
| Disease | angular_leafspot | 각진잎반점 (Angular leafspot) | 245 | 43 | 147 | 435 |
| Disease | blossom_blight | 꽃마름병 (Blossom blight) | 117 | 29 | 62 | 208 |
| Disease | leaf_spot | 잎반점 (Leaf spot) | 382 | 71 | 162 | 615 |
| Disease | powdery_mildew_leaf | 흰가루병-잎 (Powdery mildew leaf) | 319 | 63 | 151 | 533 |
| Disease | powdery_mildew_fruit | 흰가루병-과실 (Powdery mildew fruit) | 80 | 12 | 43 | 135 |
| Healthy | healthy | 정상 (Healthy foliage and fruit) | 180 | 25 | 25 | 230 |
| **Total** | **14 Classes** | **All Categories Combined** | **2,818** | **538** | **1,116** | **4,472** |

---

## 4. Quantitative Benchmark & Quantization Results

### Main Benchmark Comparison (Table 2, Test $n = 1{,}116$)

| Model Architecture | Val Acc (%) | Test Acc (%) | Macro-F1 (%) | Model Size (MB) |
|--------------------|------------:|-------------:|-------------:|----------------:|
| VGG-16 | 91.26 | 91.40 | 84.71 | 512.4 |
| VGG-19 | 91.45 | 91.40 | 84.95 | 532.6 |
| ResNet-50 | 75.46 | 75.63 | 67.54 | 89.8 |
| Inception-V3 | 91.64 | 91.85 | 85.67 | 83.2 |
| InceptionResNet-V2 | 85.50 | 85.75 | 78.43 | 207.2 |
| MobileNet | 85.32 | 85.48 | 76.54 | 8.6 |
| NASNetMobile | 85.87 | 86.02 | 77.62 | 11.9 |
| DenseNet-121 | 93.68 | 93.91 | 88.76 | 26.6 |
| Standard SKNet | 53.16 | 53.58 | 45.12 | 90.7 |
| AG-Transfer ConvNeXt (FP32) | 96.65 | 96.95 | 92.84 | 136.4 |
| AG-Transfer ConvNeXt (INT8 Proposed) | 96.84 | 97.04 | 93.22 | 34.2 |

### Quantization Ablation Study (Table 4)

| Quantization Scheme | Precision Format | Disk Size (MB) | Compression Ratio | Test Accuracy (%) |
|---------------------|------------------|---------------:|-------------------|------------------:|
| Case 1: Full Precision Baseline | FP32 | 136.5 | 1.0× | 96.95% |
| Case 2: Half Precision Optimization | FP16 | 68.3 | 2.0× | 96.95% |
| Case 3: Dynamic Quantization | Dynamic INT8 | 62.3 | 2.2× | 96.68% |
| Case 4: Weight-Packed Quantization (Selected) | Static INT8 | 34.2 | 4.0× | 97.04% |

---

## 5. Quick Start (Training & Quantization)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare strawberry_open_v13 (choose one path)
# 2a. Already split upstream (train/val/test with 14 class folders):
python scripts/prepare_dataset.py --mode copy_layout --src data/raw/strawberry_open_v13 --verify

# 2b. Merge public sources -> pooled/, then paper-fixed per-class split (seed=42):
cp configs/dataset/source_mapping.example.json data/raw/source_mapping.json
#    (edit paths, download Afzaal / IP102 / PlantVillage into data/raw/sources/)
python scripts/prepare_dataset.py --mode collect_sources --mapping data/raw/source_mapping.json
python scripts/prepare_dataset.py --mode exact --pooled-root data/raw/pooled --seed 42 --verify

# 2c. Fixed file list (reproducible manifest CSV):
python scripts/prepare_dataset.py --mode manifest --manifest data/splits/manifest.csv --verify

# 3. Verify AG-Transfer ConvNeXt architecture & exact FP32/FP16/INT8 sizes (no checkpoint required)
python models/ag_convnext.py
python scripts/quantize.py --verify-only

# 4. Train AG-Transfer ConvNeXt on strawberry_open_v13 (100 epochs, SGD Nesterov 0.9, lr=1e-4, ReduceLROnPlateau)
python experiments/train.py --model ag_convnext --pretrained --epochs 100 --batch-size 16 --lr 1e-4

# 5. Run Post-Training INT8 Weight-Packed Quantization on a trained checkpoint
python scripts/quantize.py --checkpoint results/best_ag_convnext_fp32_seed42.pt --data-dir data/processed/strawberry
```

**Note on Model Weights:** Large binary checkpoint files (`*.pt`, 34.2 MB – 136.4 MB) are excluded from Git tracking via `.gitignore`. Running `experiments/train.py` or `scripts/quantize.py` deterministically reproduces the FP32 and INT8 weight-packed checkpoints under `results/`.

---

## 6. Data Availability Statement (For Manuscript Inclusion)

Data Availability Statement: The curated 14-class strawberry pest and disease benchmark dataset (strawberry_open_v13, $N = 4{,}472$, synthesized from the open-access Strawberry Disease Detection Dataset [Afzaal et al., 2021], IP102 agricultural pest benchmark proxies, and the PlantVillage healthy strawberry repository) and the PyTorch source code for the proposed AG-Transfer ConvNeXt architecture and post-training INT8 quantization pipeline are publicly available at https://github.com/sherzod992/strawberry-advanced-danet.
