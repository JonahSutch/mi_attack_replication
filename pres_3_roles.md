# Presentation Roles — 3-Person Split

## Role 1: Data & Evaluation
**Tasks:**
- Extend to 1 additional dataset
- Tune delusion threshold for higher precision

Responsible for sourcing and integrating the new dataset into the existing pipeline, and conducting threshold analysis on attack outputs to optimize precision.

---

## Role 2: Shadow Model Scaling
**Tasks:**
- Train shadow models matching target model size
- Increase number of shadow models (~100)

Responsible for scaling up the shadow model training infrastructure — ensuring model sizes align with the target and running the expanded set of ~100 shadow models.

---

## Role 3: Attack Model Training
**Tasks:**
- Train attack model for each individual target size

Responsible for training a dedicated attack model per target model size, evaluating per-size attack performance, and synthesizing results across configurations.
