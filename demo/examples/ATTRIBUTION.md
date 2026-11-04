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
