# NIH Chest X-Ray Multi-Label Disease Classification

Multi-label classification of 14 thoracic diseases from 112,120 chest X-ray images using the NIH ChestX-Ray14 dataset. Five deep learning architectures are benchmarked — ResNet-50, VGG-19, ViT-B/16, Inception V3, and the proposed Swin Transformer V2-S — with Grad-CAM++ explainability.

---

## Results

| Model | Type | AUC | F1 | Recall |
|-------|------|-----|----|--------|
| **Swin Transformer V2-S** | Transformer (Proposed) | **0.8368** | **0.3859** | **0.5237** |
| Inception V3 | CNN (Baseline) | 0.8366 | 0.3198 | 0.4833 |
| ResNet-50 | CNN (Baseline) | 0.8270 | 0.3128 | 0.4858 |
| ViT-B/16 | Transformer (Baseline) | 0.8071 | 0.2791 | 0.4150 |
| VGG-19 | CNN (Baseline) | 0.7897 | 0.2459 | 0.4509 |

---

## Dataset

**NIH ChestX-Ray14** — Wang et al., IEEE CVPR 2017

- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 14 disease classes (multi-label — one image can have multiple diseases)
- Labels extracted via NLP from radiology reports (~90% accuracy)
- Download: [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) or [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**Place dataset at:** `D:/Data/` with subfolders `images_001/images/` to `images_012/images/`

---

## Project Structure

```
chest-xray-multilabel-classification/
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Data preparation & splits
│   ├── 03_training.ipynb        # Model training & evaluation
│   └── 04_gradcam.ipynb         # Grad-CAM++ visualization
│
├── saved_models/                 # Best model weights (.pth)
├── results/
│   └── All_Models_Results.csv   # Metrics for all models
│
└── README.md
```

---

## Key Design Decisions

**Multi-label classification** — Each image can contain multiple concurrent diseases. Uses `BCEWithLogitsLoss` with `Sigmoid` (not `CrossEntropy` + `Softmax`).

**Class imbalance handling** — `pos_weight` in `BCEWithLogitsLoss` scales the loss for rare diseases. Hernia (227 samples) receives a weight of ~493 vs No Finding (~60k samples).

**Per-class threshold optimization** — Optimal classification threshold found per disease on the validation set by maximizing F1, then applied to the test set.

**Grad-CAM++ explainability** — Applied to the Swin Transformer to highlight which X-ray regions influenced each prediction, addressing the clinical trust problem.

---

## Setup

```bash
pip install torch torchvision pytorch-grad-cam scikit-learn seaborn pandas pillow tqdm
```

Run notebooks in order: `01_EDA` → `02_preprocessing` → `03_training` → `04_gradcam`

---

## Citation

```bibtex
@InProceedings{wang2017chestxray,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald},
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {IEEE CVPR},
  pages     = {3462--3471},
  year      = {2017}
}
```