# Router, validation split (temperature and thresholds chosen here)

Temperature T = 0.5582 (ECE 0.120 -> 0.005). Thresholds: {'palette': 0.3, 'other_houseplant': 0.5, 'non_plant': 0.85}.

```json
{
 "n": 1799,
 "crop_n": 924,
 "crop_misroute_pct": 0.7575757575757576,
 "crop_misroute_n": 7,
 "top1_acc": 0.981100611450806,
 "macro_f1": 0.9691030318006623,
 "ece": 0.00453714074154015,
 "palette_routed_correctly_pct": 95.73991031390135,
 "palette_kept_out_of_crop_pct": 98.87892376681614,
 "other_houseplant_routed_correctly_pct": 98.11320754716981,
 "other_houseplant_kept_out_of_crop_pct": 98.74213836477988,
 "non_plant_routed_correctly_pct": 99.09909909909909,
 "non_plant_kept_out_of_crop_pct": 99.09909909909909,
 "green_nonplant_n": 33,
 "green_nonplant_stopped_pct": 100.0,
 "ece_before_T": 0.12042299363064723
}
```

## Routing per class

| class | n | routed correctly % | kept out of crop pipeline % |
|---|---:|---:|---:|
| bloodleaf | 44 | 97.7 | 100.0 |
| caladium | 44 | 100.0 | 100.0 |
| coleus | 45 | 93.3 | 97.8 |
| copperleaf | 44 | 95.5 | 97.7 |
| crop | 924 | 99.2 | 0.8 |
| croton | 45 | 91.1 | 97.8 |
| dieffenbachia | 45 | 95.6 | 100.0 |
| golden_pothos | 45 | 93.3 | 95.6 |
| non_plant | 111 | 99.1 | 99.1 |
| other_houseplant | 318 | 98.1 | 98.7 |
| polka_dot | 45 | 100.0 | 100.0 |
| purple_heart | 45 | 97.8 | 100.0 |
| red_cordyline | 44 | 93.2 | 100.0 |

## Confusion matrix (top-1, before thresholds)

| true \ pred | bloodleaf | caladium | coleus | copperleaf | crop | croton | dieffenbac | golden_pot | non_plant | other_hous | polka_dot | purple_hea | red_cordyl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bloodleaf | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| caladium | 0 | 44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| coleus | 0 | 0 | 42 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| copperleaf | 0 | 0 | 0 | 42 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| crop | 0 | 0 | 1 | 1 | 916 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 1 |
| croton | 0 | 0 | 0 | 0 | 1 | 41 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| dieffenbachia | 0 | 0 | 0 | 0 | 0 | 0 | 43 | 1 | 0 | 1 | 0 | 0 | 0 |
| golden_pothos | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 42 | 0 | 1 | 0 | 0 | 0 |
| non_plant | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 110 | 0 | 0 | 0 | 0 |
| other_houseplant | 0 | 0 | 1 | 0 | 4 | 0 | 0 | 1 | 0 | 312 | 0 | 0 | 0 |
| polka_dot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45 | 0 | 0 |
| purple_heart | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 44 | 0 |
| red_cordyline | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 41 |

## Model selection (validation, each with its own temperature and thresholds)

| router | input | val macro-F1 | crop moved out | ornamentals routed correctly | houseplants | non-plants | green non-plant objects stopped | Mac CPU | ONNX |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 (ImageNet-1k) | 224 px | 0.910 | 0.97 % | 83.2 % | 90.3 % | 96.4 % | 90.9 % | – | 16 MB |
| **EfficientNetV2-S (ImageNet-21k)** | 288 px | **0.969** | **0.76 %** | **95.7 %** | **98.1 %** | **99.1 %** | **100 %** | ~95 ms | 81 MB |

Chosen: EfficientNetV2-S (better on every validation number, within the 300 ms / 100 MB budget).
