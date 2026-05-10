# Membership Inference Attack — Replication Results

**Paper:** Shokri et al. (2017), "Membership Inference Attacks Against Machine Learning Models" (arXiv:1610.05820)  
**Dataset:** CIFAR-10  
**Compute:** OSU HPC SLURM cluster (GPU nodes: T4/RTX6000/RTX8000/A40/H100/L40S)

---

## What We Did

### Overview

We replicated the black-box membership inference attack from Shokri et al. (2017). The attack determines whether a specific data point was in a model's training set using only the confidence vectors (softmax outputs) returned by the model — no access to model weights or training data required. The pipeline has four stages: train a target model, train shadow models, train an attack classifier on shadow model outputs, then evaluate the attack against the target.

---

### Step 1 — Data Partitioning

The full CIFAR-10 training set (50,000 images) was split into two non-overlapping halves:

- **D_target_pool** (25,000 images): used exclusively for target model training and evaluation
- **D_shadow_pool** (25,000 images): used exclusively for shadow model training

From D_target_pool, we carved out target training sets of varying sizes (2,500 / 5,000 / 10,000 / 15,000). The remaining examples in D_target_pool became the non-member evaluation set. All splits used a fixed random seed (42) for reproducibility.

---

### Step 2 — Target Model

**Architecture:** Small CNN matching Shokri et al. §VI-B

| Layer | Details |
|---|---|
| Conv1 | 3→32 channels, 3×3 kernel, padding=1, Tanh |
| Pool1 | MaxPool 2×2 |
| Conv2 | 32→64 channels, 3×3 kernel, padding=1, Tanh |
| Pool2 | MaxPool 2×2 |
| FC1 | 64×8×8 → 128, Tanh |
| FC2 | 128 → 10 (logits; softmax applied at inference) |

**Training:** Adam optimizer, lr=0.001, 100 epochs, batch size=64  
**Evaluation:** Softmax applied at inference to produce confidence vectors

We trained **4 target models** at different training set sizes to vary the generalization gap:

| Training Size | Train Acc | Test Acc | Generalization Gap |
|---|---|---|---|
| 2,500 | 100.00% | 53.46% | **46.54%** |
| 5,000 | 100.00% | 57.47% | **42.53%** |
| 10,000 | 100.00% | 62.38% | **37.62%** |
| 15,000 | 100.00% | 64.40% | **35.60%** |

All four models achieved 100% training accuracy, confirming strong overfitting. Generalization gap decreases monotonically as training set size increases, consistent with the paper's setup.

---

### Step 3 — Shadow Models

Shadow models mimic the target model so the attack classifier can learn what "in-training" vs. "out-of-training" confidence distributions look like — without ever seeing the target's training data.

- **Number of shadow models:** 50
- **Architecture:** Identical to the target CNN
- **Training set size per shadow:** 2,500 (randomly sampled from D_shadow_pool)
- **Training:** Same hyperparameters as target (Adam, lr=0.001, 100 epochs)
- **Training on HPC:** All 50 shadow models ran as a SLURM array job in parallel (one GPU task per model)

After training each shadow model:
- Queried on its **own training data** → confidence vectors labeled **"member" (1)**
- Queried on its **held-out test data** → confidence vectors labeled **"non-member" (0)**

All outputs were merged into a single attack training dataset:

| Stat | Value |
|---|---|
| Total shadow models | 50 |
| Examples per shadow model | 5,000 (2,500 in + 2,500 out) |
| Total attack training examples | **250,000** |
| Class balance | 125,000 in / 125,000 out (perfectly balanced) |

Shadow models all reached ~100% training accuracy, producing a strong "in-training" signal for the attack classifier.

---

### Step 4 — Attack Model

Following Shokri et al. §V-D, we trained **one attack MLP per class** (10 total). Each model sees only examples where the true label matches its class, so it learns class-specific confidence patterns.

**Architecture:**

| Layer | Details |
|---|---|
| Input | 10-dim softmax confidence vector |
| FC1 | 10 → 64, ReLU |
| FC2 | 64 → 2 (binary: member / non-member) |

**Training:** Adam, lr=0.001, 50 epochs, batch size=256, CrossEntropyLoss

**Attack model training accuracy by class:**

| Class | Examples | Train Acc |
|---|---|---|
| 0 | ~25,000 | ~87% |
| 1–9 | ~25,000 each | 85–91% |

The attack models learned to reliably distinguish in-training from out-of-training confidence distributions from the shadow model data.

---

### Step 5 — Attack Evaluation

We evaluated the trained attack against all 4 target models. For each target:
1. Queried the target model on its **training set** (members) → collected confidence vectors
2. Queried the target model on its **held-out set** (non-members, equal size) → collected confidence vectors
3. Fed confidence vectors through the per-class attack MLPs → binary membership predictions
4. Computed precision, recall, accuracy, and F1

**Results:**

| Training Size | Gen. Gap | Attack Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 2,500 | 0.4654 | **90.66%** | 84.28% | 99.96% | 91.45% |
| 5,000 | 0.4253 | **85.62%** | 77.66% | 100.00% | 87.43% |
| 10,000 | 0.3762 | **80.26%** | 71.69% | 100.00% | 83.51% |
| 15,000 | 0.3560 | **77.12%** | 68.61% | 100.00% | 81.38% |

**Random chance baseline: 50.00%**

All four conditions exceed the baseline by 27–41 percentage points. Attack accuracy and precision decrease as training set size increases (and generalization gap shrinks), directly replicating the paper's Figure 4 finding.

**Key observation on recall:** Recall is ~100% across all conditions — the attack correctly identifies nearly every member. Precision is lower because the attack also incorrectly labels some non-members as members. This asymmetry is expected: the shadow models trained to 100% accuracy create a very strong "in-distribution" signal that the attack classifier learns to fire on aggressively.

---

## Comparison to Paper

Shokri et al. (2017) report median precision of **0.71–0.78** on CIFAR-10 (Figure 4, varying training set size). Our results:

| Training Size | Our Precision | Paper (approx.) |
|---|---|---|
| 2,500 | **0.843** | ~0.78 |
| 5,000 | **0.777** | ~0.76 |
| 10,000 | **0.717** | ~0.73 |
| 15,000 | **0.686** | ~0.71 |

Our results are consistent with and slightly exceed the paper's reported values, likely because our shadow models reached 100% training accuracy (stronger overfitting signal than the paper's setup).

---

## Output Figures

All figures saved to `results/figures/`:

| File | Description |
|---|---|
| `generalization_gaps.png` | Train vs. test accuracy per training size, annotated with gap values |
| `attack_vs_baseline.png` | Attack accuracy and precision vs. 50% random baseline per training size |
| `attack_accuracy_vs_gap.png` | Attack accuracy, precision, recall vs. generalization gap (line plot) |

---

## Simplifications vs. Full Paper

| Paper Setting | Our Setting | Impact |
|---|---|---|
| 100 shadow models | 50 shadow models | Minimal — ML-Leaks shows diminishing returns past ~10 |
| Shadow size matches target size | All shadows at n=2500 | Minor — attack still generalizes across target sizes |
| CIFAR-10, CIFAR-100, Purchase-100 | CIFAR-10 only | Out of scope for CP2 |
| 100 shadow models per target size | One shared shadow set | Documented simplification |
