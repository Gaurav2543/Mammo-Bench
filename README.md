# Mammo-Bench: A Large Scale Benchmark Dataset of Mammography Images

## Overview
Mammo-Bench is one of the largest open-source mammography datasets, comprising 71,844 high-quality mammographic images from 26,500 patients across 8 countries. This comprehensive dataset combines and standardizes images from seven well-curated public resources: [INbreast](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset), [Mini-DDSM](https://www.kaggle.com/datasets/cheddad/miniddsm2), [KAU-BCMD](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset), [CMMD](https://www.cancerimagingarchive.net/collection/cmmd/), [CDD-CESM](https://www.cancerimagingarchive.net/collection/cdd-cesm/), [RSNA Screening Dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data), and [DMID](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883).

## Dataset Access
The complete dataset can be accessed [here](https://datafoundation.iiit.ac.in/dataset-versions/469a02c0-de8e-4827-bf8c-14003a46b507).

## Code Availability
The preprocessing code is available in our [GitHub repository](https://github.com/Gaurav2543/Mammo-Bench).

## Dataset Structure
```
Mammo-Bench/
├── Original_Dataset/           # Raw mammography images
├── Masks/                      # Generated breast region segmentation masks
├── Preprocessed_Dataset/       # Processed images after standardization
└── CSV_Files/                  # Clinical metadata and annotations
```

## Annotations
The dataset includes comprehensive annotations in CSV format:
- BI-RADS scores (0-6)
- Breast density classifications (ACR A-D)
- Case labels (Normal/Benign/Malignant)
- Molecular subtypes (for subset)
- Abnormality types (mass, calcification, or both)

## Preprocessing Pipeline
All images have undergone:
1. Data format standardization
2. Breast segmentation using OpenBreast toolkit
3. Pectoral muscle removal
4. Intelligent cropping
5. Binary mask generation

## Usage
```python
# Example code for loading dataset
import cv2
import pandas as pd

# Load annotations
annotations = pd.read_csv('CSV_Files/mammo-bench.csv', low_memory=False)

# Loading an image
img_path = 'Preprocessed_Dataset/dataset/dataset_imageID.jpg'
image = cv2.imread(img_path)

# Loading the Binary Mask
mask_path = 'Masks/dataset.dataset_imageID.jpg'
mask = cv2.imread(mask_path)
```

## Citation
If you use this dataset in your research, please cite:
```
[Citation will be added after publication]
```

## License
This dataset is licensed under CC BY-NC-SA 4.0

## Disclaimer
This dataset is intended for research purposes only and should not be used for direct clinical diagnosis.

## Contact
For questions or issues, please contact:
- Email: gaurav.bhole@research.iiit.ac.in
- GitHub: [Mammo-Bench](https://github.com/Gaurav2543)

## Acknowledgments
We thank the original creators of INbreast, Mini-DDSM, KAU-BCMD, CMMD, CDD-CESM, RSNA Screening Dataset, and DMID for making their datasets publicly available.
