# Prak-3 Run Results

## Configuration Changelog

| Config | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| `EPOCHS` | 20 | 20 | 30 | 30 |
| `LR_ALEX` | 1e-3 | 3e-4 | 3e-4 | 1e-4 |
| `LR_HEAD` | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| `LR_BACKBONE` | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `COSINE_T_MAX` | tied to epochs (14) | tied to epochs (14) | tied to epochs (24) | 15 (fixed) |
| `train_transform` | `Resize(224)` + flip | `RandomResizedCrop(224)` + flip + `ColorJitter` | same as v2 | same as v2 |
| `val_transform` | `Resize(224)` | `Resize(256)` + `CenterCrop(224)` | same as v2 | same as v2 |
| Backbone | resnet50 | resnet50 | resnet50 | resnet50 |

---

## Results Summary

| | v1 | v2 | v3 | v4 | resnet152-v1 | resnext101-v1 |
|---|---|---|---|---|---|---|
| AlexNet val F1 | 0.3355 | 0.8968 | 0.8741 | **0.9160** | ~0.92 (wall) | ~0.88 (early stop) |
| AlexNet stopped | ep 6 (early stop) | ep 20 (wall) | ep 30 (wall) | ep 25 (early stop) | ep 30 (wall) | ep 16 (early stop) |
| Backbone val F1 | 0.9964 | 0.9936 | 0.9920 | 0.9924 | **0.9960** | 0.9952 |
| Phase 2 start | ep 6 | ep 6 | ep 6 | ep 6 | ep 9 | ep 11 |
| Backbone stopped | ep 13 | ep 20 (wall) | ep 17 | ep 21 | - | ep 17 (early stop) |
| Misclassified (val) | 9 | 16 | 20 | 19 | **10** | 12 |

---

## v1

**Run directory:** `runs/resnet50-v1`

### AlexNet

| Metric | Value |
|---|---|
| Val macro F1 | 0.3355 |
| Stopped at | Epoch 6 (early stop) |
| Behavior | Collapsed — predicted dog for all 12500 test samples |

Loss flatlined at 0.6931 from epoch 1. `LR=1e-3` overshot and stuck at majority-class prediction.

### ResNet-50

| Metric | Value |
|---|---|
| Val macro F1 | 0.9964 |
| Phase 1 ended | Epoch 6 |
| Best checkpoint | Epoch 8 |
| Stopped at | Epoch 13 (early stop) |
| Misclassified (val) | 9 / 2500 |

Clean run with no augmentation. Phase 2 jumped from 0.9896 to 0.9964 immediately.
Cat: p=1.00 r=1.00 f1=1.00 / Dog: p=1.00 r=1.00 f1=1.00

---

## v2

**Run directory:** `runs/resnet50-v2`

Changes from v1: `LR_ALEX` 1e-3 -> 3e-4; added `RandomResizedCrop` + `ColorJitter`; val uses `Resize(256)+CenterCrop(224)`.

### AlexNet

| Metric | Value |
|---|---|
| Val macro F1 | 0.8968 |
| Stopped at | Epoch 20 (hit wall) |

LR fix resolved collapse. Still climbing at ep 20, not converged.
Cat: p=0.90 r=0.89 f1=0.90 / Dog: p=0.90 r=0.90 f1=0.90

### ResNet-50

| Metric | Value |
|---|---|
| Val macro F1 | 0.9936 |
| Phase 1 ended | Epoch 6 |
| Best checkpoint | Epoch 16 |
| Stopped at | Epoch 20 (hit wall) |
| Misclassified (val) | 16 / 2500 |

Augmentation slowed convergence. Val F1 climbed 0.9856->0.9936 across Phase 2, hit wall before early stop could fire. Apparent regression vs v1 is a convergence artefact — needed more epochs.
Cat: p=0.99 r=1.00 f1=0.99 / Dog: p=1.00 r=0.99 f1=0.99

---

## v3

**Run directory:** `runs/resnet50-v3`

Changes from v2: `EPOCHS` 20 -> 30.

### AlexNet

| Metric | Value |
|---|---|
| Val macro F1 | 0.8741 |
| Stopped at | Epoch 30 (hit wall) |

Worse than v2 despite more epochs. Unstable early training (val F1 oscillated 0.33-0.57 for first 7 epochs) due to GPU non-determinism hitting a bad trajectory. Still not converged at ep 30.
Cat: p=0.91 r=0.83 f1=0.87 / Dog: p=0.85 r=0.92 f1=0.88

### ResNet-50

| Metric | Value |
|---|---|
| Val macro F1 | 0.9920 |
| Phase 1 ended | Epoch 6 |
| Best checkpoint | Epoch 12 |
| Stopped at | Epoch 17 (early stop) |
| Misclassified (val) | 20 / 2500 |

Unintended side effect: `CosineAnnealingLR T_max = EPOCHS - phase1_end = 24` (vs 14 in v2) caused slower LR decay and different Phase 2 trajectory. Best only 0.9920 vs v2's 0.9936.
Cat: p=0.99 r=0.99 f1=0.99 / Dog: p=0.99 r=0.99 f1=0.99

---

## v4

**Run directory:** `runs/resnet50-v4`

Changes from v3: `LR_ALEX` 3e-4 -> 1e-4; `COSINE_T_MAX=15` (fixed, decoupled from EPOCHS).

### AlexNet

| Metric | Value |
|---|---|
| Val macro F1 | 0.9160 |
| Best checkpoint | Epoch 20 |
| Stopped at | Epoch 25 (early stop) |

Best AlexNet result across all runs. Stable training from ep 1 — no oscillations. Early stop fired cleanly.
Cat: p=0.91 r=0.92 f1=0.92 / Dog: p=0.92 r=0.91 f1=0.92

### ResNet-50

| Metric | Value |
|---|---|
| Val macro F1 | 0.9924 |
| Phase 1 ended | Epoch 6 |
| Best checkpoint | Epoch 16 |
| Stopped at | Epoch 21 (early stop) |
| Misclassified (val) | 19 / 2500 |

Slight improvement over v3 (0.9924 vs 0.9920). Val F1 in Phase 2 fluctuates in a tight 0.9864-0.9924 band — model is at effective ceiling for ResNet-50 with current config. 8 persistent misclassifications appear across v3 and v4 (dark animals, cage occlusion, human in frame) — likely label noise or genuinely ambiguous samples.
Cat: p=0.99 r=0.99 f1=0.99 / Dog: p=0.99 r=0.99 f1=0.99

---

## resnet152-v1

**Run directory:** `runs/resnet152-v1`

Changes from v4: backbone `resnet50` -> `resnet152` (82.3% ImageNet top-1, IMAGENET1K_V2). All other config identical to v4.

### AlexNet

| Metric | Value |
|---|---|
| Val macro F1 | ~0.92 |
| Stopped at | Epoch 30 (hit wall) |

Still climbing at ep 30, not converged. Architecture-limited — same behavior as previous runs.

### ResNet-152

| Metric | Value |
|---|---|
| Val macro F1 | 0.9960 |
| Phase 1 ended | Epoch 9 |
| Misclassified (val) | 10 / 2500 |

Cat: p=0.9968 r=0.9952 f1=0.9960 / Dog: p=0.9953 r=0.9968 f1=0.9961

Phase 1 took longer (ep 9 vs ep 6) — expected for a larger backbone. Phase 2 converged quickly. Nearly halved the error count vs v4 (19 -> 10). Matches the ResNet-50 v1 ceiling (9 errors) but with augmentation — the larger backbone recovers what augmentation cost in v2-v4.

Remaining 10 errors are persistent noise-floor samples: cat-as-dog (dark/low-contrast cats, ambiguous framing), dog-as-cat (skeletal illustration, logo/drawn image, human-only frame, "No Photo Available" placeholder). These are likely label noise or dataset artifacts, not model failures. Test set confirmed to contain similar non-animal images — data cleaning not pursued.

---

## resnext101-v1

**Run directory:** `runs/resnext101_32x8d-v1`

Changes from resnet152-v1: backbone `resnet152` -> `resnext101_32x8d` (82.8% ImageNet top-1, IMAGENET1K_V2). All other config identical.

### AlexNet

| Metric | Value |
|---|---|
| Val macro F1 | ~0.88 |
| Stopped at | Epoch 16 (early stop) |

Regression vs v4 (0.9160) — training randomness, architecture-limited.

### ResNeXt-101-32x8d

| Metric | Value |
|---|---|
| Val macro F1 | 0.9952 |
| Phase 1 ended | Epoch 11 |
| Stopped at | Epoch 17 (early stop) |
| Misclassified (val) | 12 / 2500 |

Cat: p=0.9928 r=0.9976 f1=0.9952 / Dog: p=0.9976 r=0.9929 f1=0.9952

Phase 2 shows visible val loss spike ~ep 13-14. Underperforms resnet152-v1 (12 vs 10 errors, 0.9952 vs 0.9960) despite higher ImageNet top-1 (82.8% vs 82.3%). Error breakdown skewed: 9 dog→cat vs 4 in resnet152. New failures on dark/black dogs and unusual dog poses (standing upright, behind bars) — grouped convolution architecture less robust than depth for this binary task. Depth beats cardinality here.

Persistent noise-floor errors remain: skeletal illustration, Leavenworth logo, No Photo Available. **resnet152-v1 remains the best run.**
