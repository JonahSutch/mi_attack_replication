# Presentation Roles P3

## Role 1: Data & Evaluation
## Person:
## Branch: Data_&_Evaluation
**Tasks:**
- Extend to 1 additional dataset (Could be CIFAR-100 or Purchase-100)
- Tune delusion threshold for higher precision
- train at least 10 shadow models using P2 framework with new dataset

Responsible for sourcing and integrating the new dataset into the existing pipeline, and conducting threshold analysis on attack outputs to optimize precision.

---

## Role 2: Shadow Model Scaling
## Person: Dean Leon
## Branch: Shadow_Model_Scaling
**Tasks:**
- Train shadow models matching target model size
- Increase number of shadow models (~100)

Responsible for scaling up the shadow model training infrastructure — ensuring model sizes align with the target and running the expanded set of ~100 shadow models.

---

## Role 3: Attack Model Training
## Person: 
## Branch: Attack_Model_Training
**Tasks:**
- Train attack model for each individual target size for P2 tests

Responsible for training a dedicated attack model per target model size, evaluating per-size attack performance, and synthesizing results across configurations.
