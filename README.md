# Mammo-Bench: A Large Scale Benchmark Dataset of Mammography Images

## Overview
Mammo-Bench is one of the largest open-source mammography datasets, comprising 71,844 high-quality mammographic images from 26,500 patients across 8 countries. This comprehensive dataset combines and standardizes images from seven well-curated public resources: INbreast, Mini-DDSM, KAU-BCMD, CMMD, CDD-CESM, RSNA Screening Dataset, and DMID.

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
annotations = pd.read_csv('CSV_Files/mammo-bench.csv')

# Load an image
img_path = 'Preprocessed_Dataset/dataset_imageID.jpg'
image = cv2.imread(img_path)
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
- GitHub: Mammo-Bench[https://github.com/Gaurav2543/Mammo-Bench]

## Acknowledgments
We thank the original creators of INbreast, Mini-DDSM, KAU-BCMD, CMMD, CDD-CESM, RSNA Screening Dataset, and DMID for making their datasets publicly available.
