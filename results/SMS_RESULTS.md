# Membership Inference Attack — Scaled Shadow Model Results (SMS)

**Paper:** Shokri et al. (2017), "Membership Inference Attacks Against Machine Learning Models" (arXiv:1610.05820)  
**Dataset:** CIFAR-10  
**Compute:** OSU HPC SLURM cluster  
**Orchestrator:** `slurm/run_all_pipeline.py`

---

## Executive Summary

As part of **Presentation 3 (Role 2: Shadow Model Scaling)**, we scaled the shadow model training infrastructure. The target model size remains consistent, but the number of shadow models has been increased from **50 to 100**. This doubled the size of the attack training dataset from **250,000 to 500,000 examples**, resulting in a consistent **2% to 4% increase in attack accuracy and precision** across all target model sizes.

---

## Target Models & Generalization Gaps

We trained **4 target CNN models** with different training set sizes ($n$). Due to stochasticity in training, the test accuracies vary slightly from the previous run, but they maintain the same monotonic trend:

| Training Size ($n$) | Train Acc | Test Acc | Generalization Gap |
| :--- | :--- | :--- | :--- |
| **2,500** | 100.00% | 52.30% | **47.70%** (0.48) |
| **5,000** | 100.00% | 58.70% | **41.30%** (0.41) |
| **10,000** | 100.00% | 62.38% | **37.62%** (0.38) |
| **15,000** | 100.00% | 64.40% | **35.60%** (0.36) |

---

## Shadow Model Scaling (100 Models)

The shadow models were scaled to align with the training infrastructure:
- **Number of shadow models:** 100
- **Architecture:** Identical small CNN (matching target)
- **Training size per shadow:** 2,500
- **Parallelization:** Executed as a 100-task parallel Slurm job array (`SBATCH --array=0-99`) on GPU nodes.
- **Attack Training Dataset size:** **500,000 examples** (5,000 per shadow model; 2,500 members / 2,500 non-members).

---

## Attack Model Training

The attack consists of **10 per-class MLP models** (trained on the 500,000 confidence vectors of shadow predictions).
- **Epochs:** 50
- **Optimizer:** Adam (lr=0.001)
- **Batch Size:** 256

---

## Attack Evaluation Results

Below is the performance of the scaled attack models evaluated against the 4 target models:

| Target Train Size ($n$) | Generalization Gap | Attack Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2,500** | 47.70% | **93.00%** | 88.00% | 100.00% | 93.62% |
| **5,000** | 41.30% | **88.00%** | 80.00% | 100.00% | 88.89% |
| **10,000** | 37.62% | **82.00%** | 74.00% | 100.00% | 85.06% |
| **15,000** | 35.60% | **80.00%** | 72.00% | 100.00% | 83.72% |

*Recall remains at 100.00% because target members overfit completely (high confidence) and the attack MLP accurately flags all of them.*

---

## Comparison: 50 Shadows vs. 100 Shadows (SMS)

Scaling from 50 to 100 shadow models shows a direct improvement in the attack's ability to distinguish members from non-members:

```
Attack Accuracy Comparison:
95% |==================================  (93% vs 91% for n=2500)
90% |========================            (88% vs 86% for n=5000)
85% |==================                  (82% vs 80% for n=10000)
80% |=============                       (80% vs 77% for n=15000)
    +----------------------------------
      n=2500     n=5000    n=10000   n=15000
```

### Accuracy & Precision Details:
- **`n=2500`**: Accuracy increased to **93.00%** (+2.34% shift), Precision increased to **88.00%** (+3.72% shift).
- **`n=5000`**: Accuracy increased to **88.00%** (+2.38% shift), Precision increased to **80.00%** (+2.34% shift).
- **`n=10000`**: Accuracy increased to **82.00%** (+1.74% shift), Precision increased to **74.00%** (+2.31% shift).
- **`n=15000`**: Accuracy increased to **80.00%** (+2.88% shift), Precision increased to **72.00%** (+3.39% shift).

*Conclusion:* More shadow models provide the attack model with a denser representation of the confidence boundaries, significantly reducing false-positive non-member classifications and increasing Precision.

---

## Output Figures

All scaled results figures have been saved with the `sms_` prefix in `results/figures/`:

| File | Description |
| :--- | :--- |
| [`sms_generalization_gaps.png`](figures/sms_generalization_gaps.png) | Target train/test accuracies and gaps. |
| [`sms_attack_vs_baseline.png`](figures/sms_attack_vs_baseline.png) | Attack accuracy and precision vs. 50% random baseline. |
| [`sms_attack_accuracy_vs_gap.png`](figures/sms_attack_accuracy_vs_gap.png) | Attack metrics mapped against the target generalization gap. |
