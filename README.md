# Mammo-Bench: A Large Scale Benchmark Dataset of Mammography Images

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Overview
Mammo-Bench is one of the largest open-source mammography datasets, comprising 71,844 high-quality mammographic images from 26,500 patients across 8 countries. This comprehensive dataset combines and standardizes images from seven well-curated public resources: [INbreast](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset), [Mini-DDSM](https://www.kaggle.com/datasets/cheddad/miniddsm2), [KAU-BCMD](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset), [CMMD](https://www.cancerimagingarchive.net/collection/cmmd/), [CDD-CESM](https://www.cancerimagingarchive.net/collection/cdd-cesm/), [RSNA Screening Dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data), and [DMID](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883). Our work addresses the critical need for large-scale, well-annotated datasets in breast cancer detection by unifying and standardizing data from seven well-curated public resources.

## Dataset Access
The complete dataset can be accessed [here](https://datafoundation.iiit.ac.in/dataset-versions/469a02c0-de8e-4827-bf8c-14003a46b507).

## Code Availability
The preprocessing code is available in our [GitHub repository](https://github.com/Gaurav2543/Mammo-Bench).

## Key Features

### 📊 Dataset Statistics
- **Total Images**: 71,844 high-quality mammographic images
- **Patient Coverage**: 26,500 patients
- **Geographic Diversity**: Data from 8 countries
- **Source Datasets**: 7 well-curated public resources

### 📋 Comprehensive Annotations
- Case labels (Normal/Benign/Malignant)
- Breast density classifications (ACR A-D)
- BI-RADS scores (0-6)
- Molecular subtypes (Luminal A, Luminal B, HER2, TNBC)
- Abnormality types (mass, calcification, or both)

### 🔍 Preprocessing Pipeline
Our robust preprocessing pipeline ensures consistency while preserving clinically relevant features:
1. Data format standardization
2. Binary Mask Generation
3. Breast Segmentation and Pectoral Muscle Removal using [OpenBreast](https://github.com/spertuz/openbreast) toolkit
5. Intelligent Cropping to remove extraneous areas

## Repository Structure
```
Mammo-Bench/
├── README.md
├── LICENSE
├── CSV_Files/                  # Classification task CSV files
│   ├── mammo-bench_nbm_classification.csv
│   ├── mammo-bench_density_classification.csv
│   └── mammo-bench_birads_classification.csv
├── Classification_Codes/       # Implementation of classification models
│   ├── Hierarchial_Binary_Classification.py
│   ├── Multi-Class.py
│   └── Multi-Class_Minority_Augmentation.py
└── Data_Preparation/          # Data preprocessing notebooks
    ├── Clinical_Data_Preprocessing.ipynb
    ├── Image_Preprocessing.ipynb
    └── Mask_Overlap_and_Final_Segmentation.ipynb
```

## Performance Results
Our experiments demonstrate the effectiveness of the dataset:

### Classification Performance
- **Binary Classification**:
  - Normal vs Abnormal: 89.1% accuracy
  - Benign vs Malignant: 73.6% accuracy
- **Three-Class Classification**:
  - Without augmentation: 77.8% accuracy
  - With minority class augmentation: 78.8% accuracy

## Getting Started

### Dataset Access
The complete dataset can be accessed [here](https://datafoundation.iiit.ac.in/dataset-versions/469a02c0-de8e-4827-bf8c-14003a46b507).

### Installation & Dependencies
```bash
git clone https://github.com/Gaurav2543/Mammo-Bench.git
cd Mammo-Bench
pip install -r requirements.txt
```

### Basic Usage
```python
import torch
import pandas as pd
from pytorch_lightning import Trainer
from models.classifier import MammogramClassifier

# Load annotations
annotations = pd.read_csv('CSV_Files/mammo-bench.csv', low_memory=False)

# Loading an image
img_path = 'Preprocessed_Dataset/source_dataset/source_dataset_imageID.jpg'
image = cv2.imread(img_path)

# Loading the Binary Mask
mask_path = 'Masks/source_dataset/source_dataset_imageID.jpg'
mask = cv2.imread(mask_path)

# Initialize model
model = MammogramClassifier()

# Train model
trainer = Trainer(max_epochs=50, accelerator='gpu')
trainer.fit(model, train_loader, val_loader)
```

## Citation
If you use this dataset in your research, please cite:
```
[Citation will be added after publication]
```

## License
This project is licensed under CC BY-NC-SA 4.0 - see the [LICENSE](LICENSE) file for details.

## Contact
- **Gaurav Bhole** - [gaurav.bhole@research.iiit.ac.in](mailto:gaurav.bhole@research.iiit.ac.in)
- **Project Link**: [https://github.com/Gaurav2543/Mammo-Bench](https://github.com/Gaurav2543/Mammo-Bench)

## Acknowledgments
We thank the original creators of INbreast, Mini-DDSM, KAU-BCMD, CMMD, CDD-CESM, RSNA Screening Dataset, and DMID for making their datasets publicly available.

---
**Disclaimer**: This dataset is intended for research purposes only and should not be used for direct clinical diagnosis.
