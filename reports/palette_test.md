# Palette branch, TEST split of the palette species (first use; nothing tuned here)

Healthy = the web photos as labelled (not individually verified; see the review sheets).
Damaged = SYNTHETIC: real PlantSeg val lesions pasted onto the same test photos' normal tissue until they cover 5 / 10 / 20 % of the plant region (src/palette/synthetic.py).

## Overall

|                                  |   value |
|:---------------------------------|--------:|
| n_test_images                    |  442    |
| healthy_mean_pct                 |    3.25 |
| healthy_median_pct               |    0.7  |
| healthy_labelled_normal_pct      |   88.5  |
| normal_cutoff_pct                |   10    |
| palette_ms_median                |   50.1  |
| router_ms_median                 |   92.7  |
| old_pipeline_mean_infected_pct   |   28.1  |
| old_pipeline_graded_diseased_pct |   80.3  |

## Synthetic damage levels

| damage level   |   true % of leaf (median) |   predicted % (median) |   image ROC-AUC |   accuracy at 10 % |   pixel Dice |
|:---------------|--------------------------:|-----------------------:|----------------:|-------------------:|-------------:|
| 5 % of plant   |                       1.8 |                    0.9 |           0.527 |              0.495 |        0.177 |
| 10 % of plant  |                       3.1 |                    1.2 |           0.549 |              0.497 |        0.189 |
| 20 % of plant  |                       5.7 |                    1.5 |           0.568 |              0.499 |        0.19  |

## Per species (synthetic level: 10 %)

| species       |   n_test |   healthy_mean_pct |   healthy_median_pct |   healthy_labelled_normal_pct |   synthetic10_auc |   synthetic10_dice |   old_pipeline_mean_infected_pct |   old_pipeline_graded_diseased_pct |
|:--------------|---------:|-------------------:|---------------------:|------------------------------:|------------------:|-------------------:|---------------------------------:|-----------------------------------:|
| bloodleaf     |       44 |               5.49 |                 2.27 |                          81.8 |             0.56  |              0.199 |                             32.2 |                               84.1 |
| caladium      |       44 |               1.81 |                 0.55 |                          93.2 |             0.603 |              0.144 |                             24   |                               81.8 |
| coleus        |       45 |               3.1  |                 1.13 |                          86.7 |             0.519 |              0.057 |                             38.7 |                               95.6 |
| copperleaf    |       43 |               1.79 |                 0    |                          95.3 |             0.519 |              0.221 |                             25.9 |                               83.7 |
| croton        |       44 |               5.61 |                 3.9  |                          79.5 |             0.489 |              0.069 |                             31.7 |                               81.8 |
| dieffenbachia |       44 |               3.88 |                 0.74 |                          81.8 |             0.596 |              0.15  |                             27.8 |                               93.2 |
| golden_pothos |       45 |               1.18 |                 0    |                          97.8 |             0.578 |              0.18  |                             19.3 |                               57.8 |
| polka_dot     |       44 |               3.3  |                 0.59 |                          88.6 |             0.532 |              0.351 |                             38.3 |                               86.4 |
| purple_heart  |       45 |               2.72 |                 0.59 |                          91.1 |             0.59  |              0.324 |                             22.5 |                               64.4 |
| red_cordyline |       44 |               3.61 |                 0.74 |                          88.6 |             0.544 |              0.199 |                             21.1 |                               75   |

## Old crop pipeline vs palette branch on the same HEALTHY test images

The crop system assumes healthy tissue is green, so red/purple/white normal tissue is graded as lesion.

| healthy test photos                      |   n |   crop pipeline: mean infected % | crop pipeline: graded diseased   | crop pipeline: graded Severe   |   palette: mean abnormal % | palette: not 'Normal'   |
|:-----------------------------------------|----:|---------------------------------:|:---------------------------------|:-------------------------------|---------------------------:|:------------------------|
| variegated green (pothos, dieffenbachia) |  89 |                             23.5 | 75.3 %                           | 48.3 %                         |                        2.5 | 10.1 %                  |
| non-green (8 species)                    | 353 |                             29.3 | 81.6 %                           | 50.1 %                         |                        3.4 | 11.9 %                  |
| all 10 species                           | 442 |                             28.1 | 80.3 %                           | 49.8 %                         |                        3.2 | 11.5 %                  |

