# TAM-U-Net: Tumor Attention Module for Liver Tumor Segmentation

Deep learning framework for **liver tumor segmentation** using a modified U-Net architecture enhanced with a **Tumor Attention Module (TAM)**.

This project compares a standard **U-Net baseline** with a proposed **TAM-U-Net architecture** to evaluate whether tumor-focused attention improves segmentation performance.



## Project Overview

Liver tumor segmentation is an important task in medical image analysis for assisting radiologists in diagnosis and treatment planning.

This project implements:

- Baseline **U-Net segmentation model**
- **TAM-U-Net**, a modified architecture introducing a **Tumor Attention Module**
- Training and evaluation pipeline
- Visualization tools
- Comprehensive segmentation metrics

The goal is to evaluate whether tumor-focused attention improves segmentation accuracy.

## Dataset

This project uses the **Liver Tumor Segmentation (LiTS) Dataset**, which contains abdominal CT scans with corresponding liver and tumor segmentation masks.

The dataset is not included in this repository due to its large size (~30GB).

The dataset used in this project was accessed through Kaggle:

https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

Originally, the dataset comes from the **LiTS (Liver Tumor Segmentation) Challenge**, a benchmark dataset widely used in medical image segmentation research.

After downloading the dataset, place the processed files in the following structure:

data/
└── raw

After downloading, the folder structure should look similar to:

data/raw/
├── volume_pt1
├── volume_pt2
├── volume_pt3
├── volume_pt4
├── volume_pt5
└── segmentations

## Architecture

### Baseline Model

A standard **U-Net architecture** used for biomedical image segmentation.

### Proposed Model: TAM-U-Net

TAM-U-Net introduces a **Tumor Attention Module (TAM)** designed to enhance feature representations around tumor regions.

The module helps the network to:

- focus on tumor-relevant regions
- suppress background noise
- improve segmentation boundaries



## Project Structure

```
Liver-Tumor-Segmentation-TAM-UNet/
│
├── data/
│   ├── raw/
│   │   ├── volume_pt1
│   │   ├── volume_pt2
│   │   ├── volume_pt3
│   │   ├── volume_pt4
│   │   ├── volume_pt5
│   │   └── segmentations
│   │
│   ├── interim/
│   │   ├── train
│   │   ├── val
│   │   └── test
│   │
│   └── processed/
│       ├── images
│       └── masks
│
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── run_full_evaluation.py
│   ├── inference/
│   ├── explainability/
│   └── models/
│       ├── unet_baseline.py
│       └── tam_unet.py
│
├── outputs/
│   ├── metrics/
│   └── plots/
│
├── experiments/
├── models/
│
├── train.py
├── inference.py
├── preprocess.py
├── requirements.txt
└── README.md
```



## Training

To train the models run:

```
python -m src.training.train
```

This will train both the **Baseline U-Net** and **TAM-U-Net** models.



## Evaluation

Run the evaluation pipeline:
```
python -m src.evaluation.run_full_evaluation
```

This script:

- loads trained models
- runs inference on the dataset
- computes segmentation metrics
- generates comparison plots



## Evaluation Metrics

The following metrics are computed.

### Classical Metrics

- Precision
- Recall
- F1 Score
- Accuracy
- Intersection over Union (IoU)
- Dice Score

### Medical Segmentation Metrics

- Mean Dice per Slice
- Hausdorff Distance
- Surface Dice



## Results

| Model | Precision | Recall | F1 Score | IoU | Dice |
|------|------|------|------|------|------|
| Baseline U-Net | 0.9249 | 0.9411 | 0.9329 | 0.8742 | 0.9329 |
| TAM-U-Net | 0.9170 | 0.9574 | 0.9368 | 0.8811 | 0.9368 |

The **TAM-U-Net model shows slight improvements in recall and IoU**, indicating better tumor region detection.


## Visualizations

The evaluation pipeline generates:

- Dice comparison chart
- Confusion matrices
- ROC curve
- Precision–Recall curve

All plots are saved in:
```
outputs/plots/
```



## Example Prediction

The model predicts tumor regions from CT slices.

Example outputs include:

- Input CT image
- Ground truth mask
- Predicted tumor segmentation



## Research Direction

This project is part of ongoing experimentation in **medical image segmentation using deep learning and attention mechanisms**. Future work may extend this project into a **research publication or journal paper**.



## Author

**Arjun Khimta**  
B.Tech Computer Science



## Acknowledgements

This project was developed as part of experimentation in **deep learning for medical image segmentation**.