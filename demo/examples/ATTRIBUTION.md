# Example images: source and licence

These 29 images (and the 25 lesion masks) are a small stratified subset of the **official TEST splits** of
two public datasets, redistributed here for **non-commercial, academic** use with attribution. They were
never used to train the classifier, tune the pipeline or fit the learned severity correction.

- **PlantSeg** (25 diseased images + expert masks): T. Wei et al., *PlantSeg: A Large-Scale In-the-wild Dataset
  for Plant Disease Segmentation*, Zenodo v7 (doi: 10.5281/zenodo.17719108). Licence: **CC BY-NC 4.0**
  (https://creativecommons.org/licenses/by-nc/4.0/). Images are resized to a 768 px long side and masks are
  binarised (lesion = 255).
- **PlantDoc** (4 healthy-leaf images): D. Singh et al., *PlantDoc: A Dataset for Visual Plant Disease
  Detection*, CODS-COMAD 2020. Licence: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).
  Images resized to a 768 px long side.

Changes made: resizing, mask binarisation, file renaming. All rights remain with the original authors.

## Ornamental examples (palette branch)

Held-out TEST photos from iNaturalist observations (never used to fit palettes or train the router), resized to a 384 px long side. Licence and author per file (as given by iNaturalist):

- `palette/inat_413415062_760878965.jpg`: (c) fatima_juarez-paiz_10, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/413415062
- `palette/synthetic_inat_413415062_760878965.jpg`: modified from the photo above (PlantSeg lesions pasted on, CC BY-NC 4.0 parts)
- `palette/inat_410027834_754027447.jpg`: (c) bentleyroa, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/410027834
- `palette/inat_410846939_756334956.jpg`: (c) Thies, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/410846939
- `palette/inat_413369201_760786696.jpg`: (c) chelsea05, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/413369201
- `palette/inat_413530843_761122122.jpg`: (c) rohluzuali f, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/413530843
- `palette/inat_412255621_758536884.jpg`: (c) Alejandro Arriola, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/412255621
- `palette/inat_411774782_757566685.jpg`: (c) bellasherm, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/411774782
- `palette/synthetic_inat_411774782_757566685.jpg`: modified from the photo above (PlantSeg lesions pasted on, CC BY-NC 4.0 parts)
- `palette/inat_408278770_750464517.jpg`: (c) Vander Neves, some rights reserved (CC BY-NC), CC BY-NC 4.0, https://www.inaturalist.org/observations/408278770
