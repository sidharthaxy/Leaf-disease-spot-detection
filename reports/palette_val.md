# Palette branch, VALIDATION (k, thresholds and cut-offs chosen here)

Constraint: median abnormal area on healthy val leaves <= 2 %; objective: Dice + image-level ROC-AUC on synthetic damage (real PlantSeg val lesions pasted onto the same val leaves).

| species       |   k | exclude_dull   |   trim | basis   |   q_strong |   q_weak |   dark_L |   val_n |   val_n_synthetic |   healthy_median_pct |   healthy_mean_pct |   synthetic_dice |   image_auc | brown_seeds   | meets_2pct   |
|:--------------|----:|:---------------|-------:|:--------|-----------:|---------:|---------:|--------:|------------------:|---------------------:|-------------------:|-----------------:|------------:|:--------------|:-------------|
| bloodleaf     |   6 | False          |    0.3 | all     |       0.02 |     0.08 |       20 |      44 |                38 |                 1.56 |               4.71 |            0.206 |       0.575 | False         | True         |
| caladium      |   6 | False          |    0.1 | all     |       0.02 |     0.08 |       20 |      44 |                43 |                 0.11 |               1.9  |            0.146 |       0.584 | False         | True         |
| coleus        |   6 | False          |    0.3 | all     |       0.02 |     0.08 |       20 |      45 |                39 |                 0.95 |               3.44 |            0.029 |       0.542 | False         | True         |
| copperleaf    |   6 | False          |    0.3 | all     |       0.01 |     0.05 |       20 |      44 |                35 |                 0.43 |               1.62 |            0.048 |       0.506 | False         | True         |
| croton        |   6 | True           |    0.1 | all     |       0.05 |     0.15 |       20 |      45 |                37 |                 1.9  |               3.9  |            0.12  |       0.523 | False         | True         |
| dieffenbachia |   6 | False          |    0.1 | all     |       0.02 |     0.08 |       20 |      45 |                36 |                 0.38 |               1.52 |            0.102 |       0.571 | False         | True         |
| golden_pothos |   6 | False          |    0.3 | all     |       0.01 |     0.05 |       20 |      45 |                39 |                 0    |               1.19 |            0.084 |       0.636 | False         | True         |
| polka_dot     |   6 | True           |    0.1 | all     |       0.05 |     0.25 |       20 |      45 |                27 |                 1.29 |               4.88 |            0.249 |       0.546 | False         | True         |
| purple_heart  |   6 | False          |    0.1 | all     |       0.01 |     0.05 |       20 |      45 |                34 |                 1.44 |               4.52 |            0.246 |       0.615 | False         | True         |
| red_cordyline |   6 | True           |    0.1 | all     |       0.02 |     0.08 |       20 |      44 |                41 |                 0.81 |               3.25 |            0.109 |       0.548 | False         | True         |

Label cut-offs (pooled val): **Normal < 10.0 %** (= 90th percentile of the healthy val photos, so 90 % of healthy
plants read "Normal"), **Significant >= 30.0 %** (max(10, 3 x c1), a configurable assumption). Synthetic damage barely
separates from healthy photos at image level (Youden's J = 0.02), so no "best separating" cut-off exists.

## Detection vs damage size (VALIDATION, no parameters changed)

Real PlantSeg val lesions pasted onto the healthy val photos until they cover 5 / 10 / 20 % of the rough plant
region (GrabCut prior ∩ the species' normal-tissue mask), 422 photos:

| damage level | true damaged % of our leaf mask (median) | predicted abnormal % (median) | image ROC-AUC vs healthy | pixel Dice |
|---|---:|---:|---:|---:|
| 5 % of plant | 1.8 | 0.9 | 0.557 | 0.114 |
| 10 % of plant | 3.1 | 1.1 | 0.577 | 0.119 |
| 20 % of plant | 5.8 | 1.4 | 0.613 | 0.137 |

**Finding: the palette branch does not reliably detect small damage on whole-plant web photos.** The palettes
are fitted on unsegmented scene photos, so they absorb background colours (soil, bark, mulch); brown lesions
then look "normal" (e.g. golden pothos, often photographed climbing tree bark: 3.8 % true, 0.03 % predicted).
Removing brown from the palette fit (`exclude_dull`) won for only 3 species and seeding brown pixels as
abnormal (`brown_seeds`) for none: both raise the healthy false-positive area (bark and soil between leaves),
and the selection puts the 2 % constraint first. What it does reliably is NOT call healthy
coloured tissue diseased (healthy median <= 2 %), which the crop pipeline does (see the test report).
