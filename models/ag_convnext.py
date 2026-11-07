#!/usr/bin/env python3
"""
AG-Transfer ConvNeXt (Attention-Guided ConvNeXt) Model Implementation.

Matches the exact architectural specifications in the manuscript:
"Lightweight Attention-Guided ConvNeXt for Precise Early-Stage Strawberry Pest
and Disease Diagnosis on Greenhouse Mobile Edge Devices" (Malrey Lee, JBNU).

Key Architectural Components (Section 3.3 & Figure 5):
1. Backbone: ImageNet-pretrained ConvNeXt-Tiny (timm `convnext_tiny`, C=768).
2. Heterogeneous Grouped Convolutions (Split):
   - Dual parallel branches with 3x3 and 5x5 kernels across cardinal groups (G=4)
     to simultaneously capture micro-scale pests (e.g., thrips, spider mites) and
     macroscopic foliar lesions.
3. Decoupled Selective Kernel (SK) Attention Module (Figure 5):
   - Unlike canonical SKNet (which performs element-wise summation prior to GAP),
     each branch output U_1 (3x3) and U_2 (5x5) undergoes Global Average Pooling
     independently: s_1 = GAP(U_1), s_2 = GAP(U_2) in R^C.
   - Shared Linear Projection (C -> C/r, r=16) WITHOUT Batch Normalization (BN)
     to prevent artificial compression of fine-grained lesion activation sensitivity.
   - Independent Branch Dropout (p=0.2) applied after the shared projection in
     each branch to enforce functional orthogonality and prevent feature homogenization.
   - Independent branch expansion layers (C/r -> C) followed by channel-wise Softmax
     to obtain dynamic attention weights a, b (a + b = 1), yielding V = a * U_1 + b * U_2.
4. Weakly-Dense Cross-Channel Refinement Head:
   - 1x1 cross-channel projection with local skip connection (avoiding DenseNet-style
     long-range channel explosion) -> Global Average Pooling -> 14-class linear head.
   - Total Parameters: ~34.12M (~136.4 MB in FP32, ~68.3 MB in FP16, ~34.2 MB in INT8).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


STRAWBERRY_14_CLASSES: List[str] = [
    "angular_leafspot",        # Disease 1: 각진잎반점 (435)
    "anthracnose_fruit_rot",   # Disease 2: 탄저병-과실 (97)
    "blossom_blight",          # Disease 3: 꽃마름병 (208)
    "gray_mold",               # Disease 4: 잿빛곰팡이병 (477)
    "healthy",                 # Healthy: 정상 (230)
    "leaf_spot",               # Disease 5: 잎반점 (615)
    "pest_aphid",              # Pest 1: 진딧물 (876)
    "pest_armyworm",           # Pest 2: 담배거세미나방 / Cotton leafworm (206)
    "pest_plant_bug",          # Pest 3: 노린재 (348)
    "pest_spider_mite",        # Pest 4: 점박이응애 (160)
    "pest_thrips",             # Pest 5: 총채벌레 (47)
    "pest_whitefly",           # Pest 6: 온실가루이 (105)
    "powdery_mildew_fruit",    # Disease 6: 흰가루병-과실 (135)
    "powdery_mildew_leaf",     # Disease 7: 흰가루병-잎 (533)
]


class HeterogeneousGroupConv(nn.Module):
    """
    Split stage: Parallel heterogeneous grouped convolutions (3x3 and 5x5)
    within cardinal groups to capture multi-scale pest and disease symptoms.
    """

    def __init__(
        self,
        channels: int,
        kernels: Tuple[int, int] = (3, 5),
        groups: int = 4,
    ) -> None:
        super().__init__()
        self.kernels = kernels
        self.branch_3x3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernels[0],
            padding=kernels[0] // 2,
            groups=groups,
            bias=False,
        )
        self.branch_5x5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernels[1],
            padding=kernels[1] // 2,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u1 = self.branch_3x3(x)
        u2 = self.branch_5x5(x)
        return u1, u2


class DecoupledSKAttention(nn.Module):
    """
    Decoupled Selective Kernel (SK) Attention Module (Section 3.3, Figure 5).

    Data flow (strictly matching Figure 5 in the paper):
      1. Split: F -> 3x3 Group Conv (U_1) and 5x5 Group Conv (U_2)
      2. Decoupled GAP: s_1 = GAP(U_1) in R^C, s_2 = GAP(U_2) in R^C
         (No element-wise summation before the shared projection layer)
      3. Shared Linear Layer (C -> C/r, r=16) with NO Batch Normalization (BN)
         to preserve fine-grained lesion sensitivity.
      4. Independent Branch Dropout (p=0.2) applied separately to each branch
         after the shared linear projection.
      5. Independent Branch Expansion (C/r -> C) + Channel-wise Softmax:
         a, b = Softmax([z_1, z_2], dim=1) where a + b = 1
      6. Attention Fusion: V = a * U_1 + b * U_2
    """

    def __init__(
        self,
        channels: int = 768,
        reduction: int = 16,
        groups: int = 4,
        dropout_p: float = 0.2,
        use_bn_ablation: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        reduced_dim = max(channels // reduction, 16)

        # 1. Heterogeneous multi-scale split branches (3x3 and 5x5)
        self.split_conv = HeterogeneousGroupConv(
            channels=channels, kernels=(3, 5), groups=groups
        )

        # 2. Shared linear projection (C -> C/r) WITHOUT Batch Normalization
        self.shared_fc = nn.Linear(channels, reduced_dim, bias=True)
        # Optional BN only for ablation comparison ("with BN" vs proposed "no BN")
        self.ablation_bn = nn.BatchNorm1d(reduced_dim) if use_bn_ablation else None
        self.act = nn.ReLU(inplace=True)

        # 3. Independent branch-level dropout (p = 0.2) for branch orthogonality
        self.dropout_branch1 = nn.Dropout(p=dropout_p)
        self.dropout_branch2 = nn.Dropout(p=dropout_p)

        # 4. Decoupled branch attention heads (C/r -> C)
        self.fc_branch1 = nn.Linear(reduced_dim, channels, bias=True)
        self.fc_branch2 = nn.Linear(reduced_dim, channels, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # Step 1: Heterogeneous group convolutions -> U_1 (3x3), U_2 (5x5)
        u1, u2 = self.split_conv(x)

        # Step 2: Independent Global Average Pooling per branch (Decoupled GAP)
        s1 = F.adaptive_avg_pool2d(u1, 1).view(b, c)  # [B, C]
        s2 = F.adaptive_avg_pool2d(u2, 1).view(b, c)  # [B, C]

        # Step 3: Shared Linear Layer (C -> C/r, no BN by default)
        h1 = self.shared_fc(s1)
        h2 = self.shared_fc(s2)
        if self.ablation_bn is not None:
            h1 = self.ablation_bn(h1)
            h2 = self.ablation_bn(h2)
        h1 = self.act(h1)
        h2 = self.act(h2)

        # Step 4: Independent Branch Dropout (p = 0.2)
        h1 = self.dropout_branch1(h1)
        h2 = self.dropout_branch2(h2)

        # Step 5: Independent branch projections & Softmax (a + b = 1)
        logits1 = self.fc_branch1(h1)  # [B, C]
        logits2 = self.fc_branch2(h2)  # [B, C]
        attn_logits = torch.stack([logits1, logits2], dim=1)  # [B, 2, C]
        attn_weights = self.softmax(attn_logits).unsqueeze(-1).unsqueeze(-1)  # [B, 2, C, 1, 1]

        weight_a = attn_weights[:, 0]  # [B, C, 1, 1]
        weight_b = attn_weights[:, 1]  # [B, C, 1, 1]

        # Step 6: Attention Fusion V = a * U_1 + b * U_2
        v = weight_a * u1 + weight_b * u2
        return v


class WeaklyDenseRefinementHead(nn.Module):
    """
    Weakly-Dense 1x1 Cross-Channel Refinement Block (Section 2.2 & 3.3).

    Propagates only immediate preceding feature maps via local 1x1 cross-channel
    convolutions and residual skip connections, avoiding DenseNet's quadratic
    concatenation overhead while refining localized lesion representations.
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 14,
        groups: int = 4,
        dropout_p: float = 0.2,
        use_bn_ablation: bool = False,
    ) -> None:
        super().__init__()
        self.dsk_attention = DecoupledSKAttention(
            channels=in_channels,
            reduction=16,
            groups=groups,
            dropout_p=dropout_p,
            use_bn_ablation=use_bn_ablation,
        )
        # Consecutive 1x1 cross-channel refinement (Weakly-Dense transition)
        self.pw_refine1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw_refine2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.head_dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Decoupled SK Attention with local residual skip
        v = self.dsk_attention(x) + x
        # Weakly-dense 1x1 channel refinement with immediate skip connection
        refined = self.pw_refine2(self.act(self.pw_refine1(v))) + v
        return refined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.forward_features(x)
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        pooled = self.norm(pooled)
        pooled = self.head_dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class AGTransferConvNeXt(nn.Module):
    """
    Proposed Attention-Guided ConvNeXt (AG-Transfer ConvNeXt) for 14-class
    strawberry pest and disease classification.

    - FP32 Parameter Footprint: ~34.12M parameters (~136.4 MB)
    - Static INT8 Weight-Packed Footprint: ~34.2 MB (4.0x compression)
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        backbone_name: str = "convnext_tiny",
        groups: int = 4,
        dropout_p: float = 0.2,
        use_bn_ablation: bool = False,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("Please install `timm` (`pip install timm>=0.9.0`).")

        # Load ImageNet-pretrained ConvNeXt-Tiny feature extractor (C=768 before pooling)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        in_channels = getattr(self.backbone, "num_features", 768)

        # Proposed Attention-Guided Classification Head
        self.ag_head = WeaklyDenseRefinementHead(
            in_channels=in_channels,
            num_classes=num_classes,
            groups=groups,
            dropout_p=dropout_p,
            use_bn_ablation=use_bn_ablation,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        refined = self.ag_head.forward_features(feats)
        return refined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        logits = self.ag_head(feats)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the terminal depthwise/refinement convolutional layer for Grad-CAM."""
        return self.ag_head.pw_refine2


def create_benchmark_model(
    model_name: str = "ag_convnext",
    num_classes: int = 14,
    pretrained: bool = True,
    dropout_p: float = 0.2,
    use_bn_ablation: bool = False,
) -> nn.Module:
    """
    Factory function supporting the proposed AG-Transfer ConvNeXt and all 9
    benchmark baseline models reported in Table 2 of the manuscript.
    """
    name = model_name.lower().strip()
    if name in ("ag_convnext", "ag_transfer_convnext", "proposed"):
        return AGTransferConvNeXt(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            use_bn_ablation=use_bn_ablation,
        )

    if timm is None:
        raise ImportError("Please install `timm` (`pip install timm>=0.9.0`).")

    # Table 2 Benchmark Baselines
    baseline_map: Dict[str, str] = {
        "vgg16": "vgg16",
        "vgg19": "vgg19",
        "resnet50": "resnet50",
        "inceptionv3": "inception_v3",
        "inceptionresnetv2": "inception_resnet_v2",
        "mobilenet": "mobilenetv2_100",
        "nasnetmobile": "nasnetalarge",
        "densenet121": "densenet121",
        "sknet": "skresnet34",
        "convnext_tiny": "convnext_tiny",
    }
    timm_arch = baseline_map.get(name, name)
    return timm.create_model(timm_arch, pretrained=pretrained, num_classes=num_classes)


if __name__ == "__main__":
    model = create_benchmark_model("ag_convnext", num_classes=14, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    total_params = sum(p.numel() for p in model.parameters())
    fp32_mb = (total_params * 4) / (1024 * 1024)
    int8_mb = (total_params * 1) / (1024 * 1024)
    print(f"AG-Transfer ConvNeXt Output Shape : {tuple(out.shape)}")
    print(f"Total Parameters                  : {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Estimated Size (FP32 / INT8)      : {fp32_mb:.1f} MB -> {int8_mb:.1f} MB (4.0x compression)")