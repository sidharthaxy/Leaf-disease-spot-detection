# Router, TEST split: routing-only evaluation

**Crop rows: PlantSeg/PlantDoc test split, third use, routing only, nothing tuned on it.** Temperature and thresholds come from validation (reports/router_val.md). The ornamental, houseplant and non-plant test images are held-out web images (first use).

```json
{
 "n": 2520,
 "crop_n": 1649,
 "crop_misroute_pct": 0.8489993935718617,
 "crop_misroute_n": 14,
 "top1_acc": 0.9746031746031746,
 "macro_f1": 0.9347899535207005,
 "ece": 0.007422100591038658,
 "palette_routed_correctly_pct": 91.4027149321267,
 "palette_kept_out_of_crop_pct": 98.19004524886877,
 "other_houseplant_routed_correctly_pct": 96.85534591194968,
 "other_houseplant_kept_out_of_crop_pct": 98.74213836477988,
 "non_plant_routed_correctly_pct": 94.5945945945946,
 "non_plant_kept_out_of_crop_pct": 96.3963963963964,
 "green_nonplant_n": 41,
 "green_nonplant_stopped_pct": 90.2439024390244
}
```

## Routing per class

| class | n | routed correctly % | kept out of crop pipeline % |
|---|---:|---:|---:|
| bloodleaf | 44 | 95.5 | 100.0 |
| caladium | 44 | 97.7 | 100.0 |
| coleus | 45 | 88.9 | 95.6 |
| copperleaf | 43 | 95.3 | 97.7 |
| crop | 1649 | 99.2 | 0.8 |
| croton | 44 | 81.8 | 97.7 |
| dieffenbachia | 44 | 88.6 | 100.0 |
| golden_pothos | 45 | 84.4 | 95.6 |
| non_plant | 111 | 94.6 | 96.4 |
| other_houseplant | 318 | 96.9 | 98.7 |
| polka_dot | 44 | 88.6 | 97.7 |
| purple_heart | 45 | 95.6 | 97.8 |
| red_cordyline | 44 | 97.7 | 100.0 |

## Confusion matrix (top-1, before thresholds)

| true \ pred | bloodleaf | caladium | coleus | copperleaf | crop | croton | dieffenbac | golden_pot | non_plant | other_hous | polka_dot | purple_hea | red_cordyl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bloodleaf | 42 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| caladium | 0 | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| coleus | 1 | 0 | 40 | 1 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| copperleaf | 0 | 0 | 0 | 41 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| crop | 0 | 0 | 2 | 0 | 1634 | 1 | 0 | 1 | 2 | 9 | 0 | 0 | 0 |
| croton | 0 | 0 | 2 | 0 | 1 | 36 | 0 | 0 | 0 | 3 | 0 | 1 | 1 |
| dieffenbachia | 0 | 0 | 0 | 0 | 0 | 0 | 39 | 3 | 0 | 2 | 0 | 0 | 0 |
| golden_pothos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 38 | 0 | 5 | 1 | 0 | 1 |
| non_plant | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 109 | 2 | 0 | 0 | 0 |
| other_houseplant | 0 | 3 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 309 | 2 | 0 | 0 |
| polka_dot | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 1 | 39 | 0 | 0 |
| purple_heart | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 43 | 1 |
| red_cordyline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 43 |

## Crop test images moved out of the crop pipeline (14)

The crop pipeline itself is unchanged, so every other crop test image gets byte-identical results; only these images would differ from the published numbers.

- `images/crop/test/banana_black_leaf_streak_Bing_0010.jpg` -> other_houseplant (router: other houseplant (p=0.99))
- `images/crop/test/banana_bunchy_top_Baidu_0084.jpg` -> other_houseplant (router: other houseplant (p=0.54))
- `images/crop/test/banana_bunchy_top_Bing_0105.jpg` -> other_houseplant (router: other houseplant (p=0.94))
- `images/crop/test/banana_bunchy_top_Bing_0152.jpg` -> other_houseplant (router: other houseplant (p=0.98))
- `images/crop/test/blueberry_botrytis_blight_Bing_0015.jpg` -> other_houseplant (router: other houseplant (p=0.93))
- `images/crop/test/cabbage_black_rot_Bing_0195.jpg` -> palette (router: golden_pothos (p=0.86))
- `images/crop/test/carrot_cavity_spot_google_0152.jpg` -> out_of_scope (router: not a plant (p=0.95))
- `images/crop/test/cherry_powdery_mildew_76.jpg` -> other_houseplant (router: other houseplant (p=0.72))
- `images/crop/test/citrus_greening_disease_48.jpg` -> palette (router: croton (p=0.99))
- `images/crop/test/cucumber_bacterial_wilt_109.jpg` -> other_houseplant (router: other houseplant (p=0.89))
- `images/crop/test/plum_pocket_disease_google_0117.jpg` -> other_houseplant (router: other houseplant (p=0.59))
- `images/crop/test/squash_powdery_mildew_google_0327.jpg` -> other_houseplant (router: other houseplant (p=0.73))
- `images/crop/test/tomato_mosaic_virus_143.jpg` -> palette (router: coleus (p=0.54))
- `images/crop/test/tomato_yellow_leaf_curl_virus_40.jpg` -> palette (router: coleus (p=0.54))

## Effect on the published crop test metrics

All 14 moved images are diseased PlantSeg photos (1,554 in the mask evaluation). Dropping them from the round-1 per-image results changes every metric by at most 0.0012 Dice and 0.09 pp MAE:

| experiment (round-1 test eval)   |   Dice all |   Dice without moved |   MAE all (pp) |   MAE without moved (pp) |
|:---------------------------------|-----------:|---------------------:|---------------:|-------------------------:|
| aware                            |     0.337  |               0.3375 |          19.28 |                    19.25 |
| generic                          |     0.3215 |               0.3227 |          25.85 |                    25.79 |
| predicted_nofallback             |     0.3343 |               0.3348 |          19.24 |                    19.2  |
| method_hsv                       |     0.3268 |               0.328  |          23.42 |                    23.4  |
| method_lab                       |     0.3099 |               0.3104 |          18.75 |                    18.76 |
| method_kmeans                    |     0.1979 |               0.1985 |          24.75 |                    24.75 |
| method_delta_e                   |     0.3413 |               0.342  |          51.59 |                    51.5  |

## Green-tissue gate alone on the same non-plant test photos (for comparison)

| non-plant test photos | gate "out of scope" (stopped) | gate "low green" (warning, still analysed) | gate "ok" |
|---|---:|---:|---:|
| green objects (Wikimedia, 41) | 1 | 2 | 38 |
| COCO scenes (70) | 43 | 26 | 1 |
