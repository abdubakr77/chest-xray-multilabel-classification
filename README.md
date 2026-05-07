# NIH Chest X-Ray Multi-Label Disease Classification

Multi-label classification of 14 thoracic diseases from 112,120 chest X-ray images using the NIH ChestX-Ray14 dataset. Six deep learning architectures are benchmarked — ResNet-50, VGG-19, DenseNet-121, Inception V3, ViT-B/16, and the proposed Swin Transformer V2-B — with Grad-CAM++ explainability.

---

## Results

| Model | AUC | Loss | F1 | Precision | Recall | Best Epochs | Time (Hours) |
|------|-----|------|----|-----------|--------|-------------|--------------|
| DenseNet-121 | 0.8163 | 0.0373 | 0.2816 | 0.2526 | 0.3289 | 7–15 | 3.67 |
| ResNet-50 | 0.8134 | 0.0380 | 0.2748 | 0.2440 | 0.3326 | 7 | 3.74 |
| Inception V3 | 0.8216 | 0.0373 | 0.2280 | 0.2100 | 0.2999 | 7 | 2.85 |
| Swin Transformer | 0.8505 | 0.0358 | 0.3107 | 0.2876 | 0.3662 | 12–20 | 5.83 |
| Vision Transformer (ViT-B/16) | 0.8312 | 0.0370 | 0.2931 | 0.2915 | 0.3479 | 7 | 2.57 |
| VGG-19 | 0.8341 | 0.0365 | 0.3045 | 0.2751 | 0.3664 | 7 | 3.45 |

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

**Multi-label classification** — Each image can contain multiple concurrent diseases. Uses `Focal Loss` over logits with `Sigmoid` activation (not `CrossEntropy` + `Softmax`).

**Class imbalance handling** — `Focal Loss` (α=1, γ=2) downweights easy examples and forces the model to focus on harder and rarer disease classes.

**Per-class threshold optimization** — Optimal classification threshold found independently per disease on the validation set by maximizing F1, then applied unchanged to the test set.

**Mixed precision training** — `autocast` + `GradScaler` for faster GPU utilization without sacrificing numerical stability.

**OneCycleLR scheduling** — Warm-up then cosine decay, stepped once per epoch for stable convergence.

**Early stopping** — Monitors validation AUC with patience of 5 epochs; best weights are saved and reloaded for evaluation.

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