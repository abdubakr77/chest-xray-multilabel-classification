# Swin Transformer Based Multi-Label Chest Disease Classification

[![Paper](https://img.shields.io/badge/Preprint-Research%20Square-8A2BE2)](https://www.researchsquare.com/article/rs-10932750/latest)
[![DOI](https://img.shields.io/badge/DOI-10.21203%2Frs.3.rs--10932750%2Fv1-blue)](https://doi.org/10.21203/rs.3.rs-10932750/v1)
[![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-2E8B57)](https://nihcc.app.box.com/v/ChestXray-NIHCC)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and experiment notebooks for the preprint:

> **Swin Transformer Based Multi-Label Chest Disease Classification**

This project investigates automated multi-label thoracic disease classification from chest X-ray images. Six deep learning architectures are benchmarked under the same experimental protocol on the NIH ChestX-ray14 dataset, with Grad-CAM++ analysis used to improve model interpretability.

**Paper:** [Read the preprint on Research Square](https://www.researchsquare.com/article/rs-10932750/latest)

**DOI:** [10.21203/rs.3.rs-10932750/v1](https://doi.org/10.21203/rs.3.rs-10932750/v1)

**Code:** [abdubakr77/chest-xray-multilabel-classification](https://github.com/abdubakr77/chest-xray-multilabel-classification)

> This work is currently available as a preprint and has not yet completed peer review.

## Abstract

Chest radiography is widely used for screening and diagnosing thoracic diseases, but interpreting large collections of X-ray images remains time-consuming and can be affected by class imbalance and overlapping findings. This study presents a multi-label deep learning pipeline for classifying 14 thoracic diseases from the NIH ChestX-ray14 dataset.

We benchmark ResNet-50, VGG-19, DenseNet-121, Inception V3, Vision Transformer B/16, and Swin Transformer V2-B under identical training and evaluation conditions. The proposed Swin Transformer approach achieved the strongest overall performance, reaching a reported AUC of **0.8495**. Patient-level splitting was used to reduce data leakage, Focal Loss was used to address severe class imbalance, and Grad-CAM++ was used to inspect the image regions influencing model predictions.

## Key Findings

- **Best overall model:** Swin Transformer V2-B
- **Reported best AUC:** 0.8495
- **Dataset:** NIH ChestX-ray14
- **Images:** 112,120 frontal chest X-rays
- **Patients:** 30,805 unique patients
- **Disease labels:** 14 thoracic disease categories
- **Learning setup:** Multi-label classification
- **Explainability:** Grad-CAM++ visualizations
- **Data split:** Patient-level separation to reduce leakage

## Benchmark Results

The table below summarizes the results reported in the accompanying experiments.

| Model | AUC | Loss | F1 | Precision | Recall | Best Epochs | Time |
|:--|--:|--:|--:|--:|--:|:--:|--:|
| DenseNet-121 | 0.8163 | 0.0373 | 0.2816 | 0.2526 | 0.3289 | 7 to 15 | 3.67 h |
| ResNet-50 | 0.8134 | 0.0380 | 0.2748 | 0.2440 | 0.3326 | 7 | 3.74 h |
| Inception V3 | 0.8216 | 0.0373 | 0.2280 | 0.2100 | 0.2999 | 7 | 2.85 h |
| Swin Transformer V2-B | **0.8495** | **0.0358** | **0.3107** | **0.2876** | **0.3662** | 12 to 20 | 5.83 h |
| Vision Transformer B/16 | 0.8312 | 0.0370 | 0.2931 | 0.2915 | 0.3479 | 7 | 2.57 h |
| VGG-19 | 0.8341 | 0.0365 | 0.3045 | 0.2751 | 0.3664 | 7 | 3.45 h |

These values are reported for research comparison and should be interpreted in the context of the preprint protocol, dataset, preprocessing, and evaluation thresholds.

## Methodology

### Multi-label formulation

Each X-ray can contain more than one disease. The model therefore predicts an independent probability for each disease using sigmoid activations. Multiclass softmax classification is not used.

### Patient-level splitting

Images from the same patient can be highly correlated. The preprocessing workflow uses the official NIH patient-level split lists before creating the training and validation partitions. This helps reduce patient overlap between training and evaluation data.

### Imbalance-aware optimization

The dataset contains a large number of normal studies and substantially fewer examples for rare diseases. Focal Loss with $\alpha = 1$ and $\gamma = 2$ reduces the influence of easy examples and focuses optimization on harder cases.

### Threshold optimization

A single threshold of 0.5 is not assumed to be optimal for every disease. The validation set is used to select a separate F1-maximizing threshold for each class. Those thresholds are then kept fixed when evaluating the test set.

### Training strategy

The experiments use pretrained architectures, data augmentation, mixed-precision training, AdamW optimization, OneCycle learning-rate scheduling, validation monitoring, and early stopping. The best checkpoint is selected using validation performance.

### Explainability

Grad-CAM++ is applied to the Swin Transformer model to visualize the regions that contribute to disease predictions. This provides a qualitative interpretability layer and helps assess whether predictions focus on clinically meaningful areas of the radiograph.

## Dataset

The experiments use the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC), introduced by Wang et al. The dataset contains:

- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 14 thoracic disease labels
- Single-label and multi-label findings
- Metadata including patient identity, view position, age, gender, and follow-up number

The dataset is not included in this repository. Download it from the NIH source or the [Kaggle mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data).

Expected local layout:

```text
D:/Data/
├── Data_Entry_2017.csv
├── train_val_list.txt
├── test_list.txt
├── images_001/images/
├── images_002/images/
├── ...
└── images_012/images/
```

## Repository Structure

```text
chest-xray-multilabel-classification/
├── dataset/                         # Dataset-related files and metadata
├── notebooks/
│   ├── 01_EDA.ipynb                 # Dataset exploration and label analysis
│   ├── 02_preprocessing.ipynb       # Patient-level splits and preprocessing artifacts
│   ├── 03_training.ipynb            # Model training, validation, and benchmarking
│   ├── 04_gradcam.ipynb             # Grad-CAM++ explainability analysis
│   └── data_essentials.pth          # Generated preprocessing artifact
├── results/
│   └── All_Models_Results.csv       # Benchmark metrics
├── saved_models/                    # Trained model checkpoints
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abdubakr77/chest-xray-multilabel-classification.git
cd chest-xray-multilabel-classification
```

### 2. Install dependencies

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install torch torchvision pytorch-grad-cam scikit-learn seaborn pandas pillow tqdm matplotlib jupyter
```

### 3. Prepare the dataset

Download NIH ChestX-ray14 and place the metadata files and image folders under `D:/Data/`, or update the dataset paths in the notebooks to match your local machine.

### 4. Run the notebooks in order

```text
01_EDA.ipynb
      ↓
02_preprocessing.ipynb
      ↓
03_training.ipynb
      ↓
04_gradcam.ipynb
```

The preprocessing notebook creates `data_essentials.pth`. The training notebook uses that artifact to build the data loaders, train the benchmark models, calculate validation and test metrics, and save checkpoints. The Grad-CAM++ notebook loads a trained Swin Transformer checkpoint for visual analysis.

## Reproducibility Notes

For a meaningful comparison, keep the following consistent:

- Patient-level train, validation, and test separation
- Disease label order across preprocessing, training, and evaluation
- Image normalization and model-specific input resolution
- Focal Loss configuration
- Validation-based per-class thresholds
- Model checkpoint selection criterion
- Random seeds and hardware configuration

Training six large pretrained architectures can require substantial GPU memory and several hours. The notebooks are designed for research experimentation rather than low-resource inference.

## Responsible Use

This repository is intended for research and educational use. It is not a clinical diagnostic system and must not be used to make medical decisions. Performance on NIH ChestX-ray14 may not generalize to other hospitals, populations, scanners, acquisition protocols, or clinical workflows. External validation and clinical review are required before any real-world deployment.

## Citation

If this work is useful in your research, please cite the preprint using the title and DOI below:

> *Swin Transformer Based Multi-Label Chest Disease Classification.* Research Square preprint. DOI: [10.21203/rs.3.rs-10932750/v1](https://doi.org/10.21203/rs.3.rs-10932750/v1)

The NIH ChestX-ray14 dataset was introduced in:

```bibtex
@inproceedings{wang2017chestxray8,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald},
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {3462--3471},
  year      = {2017}
}
```

## Acknowledgements

Grateful acknowledgements to Dr. Ashraf Ullah and Abdul Jabbar for their support throughout this work.

## Contact and Links

- **Preprint:** https://www.researchsquare.com/article/rs-10932750/latest
- **DOI:** https://doi.org/10.21203/rs.3.rs-10932750/v1
- **Code:** https://github.com/abdubakr77/chest-xray-multilabel-classification

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Keywords

`Machine Learning` `Deep Learning` `Medical Imaging` `Computer Vision` `Chest X-ray` `Multi-Label Classification` `Swin Transformer` `Explainable AI` `Grad-CAM++`
