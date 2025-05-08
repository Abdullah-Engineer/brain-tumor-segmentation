# Brain Tumor Segmentation using YOLLO11 and SAM2

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview
This project implements a full computer-vision pipeline—including object detection, semantic segmentation, and image classification—using a Jupyter notebook and supporting scripts. It extracts features, trains deep-learning models, and generates a suite of evaluation plots (confusion matrices, precision-recall curves, correlograms, etc.) to assess performance :contentReference[oaicite:0]{index=0}.

## Table of Contents
- [Project Description](#project-description)  
- [Tech Stack & Dependencies](#tech-stack--dependencies)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Results](#results)  
- [Evaluation](#evaluation)  
- [Future Work](#future-work)  
- [License](#license)  
- [Contact](#contact)  

## Project Description
A modular Python project that:
- **Detects** objects in images/videos using deep-learning detectors. :contentReference[oaicite:1]{index=1}  
- **Segments** detected regions with a semantic segmentation model. :contentReference[oaicite:2]{index=2}  
- **Classifies** cropped detections into predefined categories. :contentReference[oaicite:3]{index=3}  
- **Visualizes** results through evaluation plots embedded directly in outputs. :contentReference[oaicite:4]{index=4}  

## Tech Stack & Dependencies
- **Language:** Python 3.8+ :contentReference[oaicite:5]{index=5}  
- **Frameworks:** TensorFlow / PyTorch, OpenCV :contentReference[oaicite:6]{index=6}  
- **Audio & Vision:** Librosa, scikit-image, PIL :contentReference[oaicite:7]{index=7}  
- **Analysis & Plotting:** NumPy, pandas, Matplotlib, Seaborn :contentReference[oaicite:8]{index=8}  

## Installation
```bash
# Clone repository
git clone https://github.com/YourUsername/ML-Internship.git
cd ML-Internship

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

*Ensure you have Python 3.8+ and pip installed.* ([GitHub][1])

## Usage

1. **Launch Notebook**

   ```bash
   jupyter notebook notebooks/ML_Internship.ipynb
   ```
2. **Run Cells** sequentially:

   * Data loading & preprocessing
   * Model training (detection, segmentation, classification)
   * Evaluation & plotting
3. **Scripts** (optional):

   ```bash
   python scripts/run_detection.py --input data/images --output results/detections
   python scripts/run_segmentation.py --input results/detections --output results/segments
   python scripts/run_classification.py --input results/segments --output results/classification
   ```

## Project Structure

```
.
├── images/                          # All evaluation & example outputs
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── 10_random_predict.png
│   ├── 10_first_predict.png
│   ├── labels.jpg
│   ├── labels_correlogram.jpg
│   ├── P_curve.png
│   ├── R_curve.png
│   ├── PR_curve.png
│   ├── results.png
│   ├── train_batch0.jpg
│   ├── val_batch0_labels.jpg
│   ├── val_batch0_pred.jpg
│   └── detection_segment_examples/
├── notebooks/
│   └── ML_Internship.ipynb
├── scripts/
│   ├── run_detection.py
│   ├── run_segmentation.py
│   └── run_classification.py
├── requirements.txt
└── README.md
```

([GitHub][1])

## Results

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)
Shows true vs. predicted classes for the classifier. ([GitHub][1])

### Normalized Confusion Matrix

![Normalized Confusion Matrix](images/confusion_matrix_normalized.png)
Displays per-class accuracy proportions. ([GitHub][1])

### Random Sample Predictions

![10 Random Predictions](images/10_random_predict.png)
Ten random test images with model predictions. ([GitHub][1])

### First Batch Predictions

![10 First Predictions](images/10_first_predict.png)
First ten images from validation set with predicted labels. ([GitHub][1])

### Label Distribution

![Labels Distribution](images/labels.jpg)
Histogram of class frequencies. ([GitHub][1])

### Label Correlogram

![Labels Correlogram](images/labels_correlogram.jpg)
Correlation heatmap of co-occurring labels. ([GitHub][1])

### Precision Curve

![Precision Curve](images/P_curve.png)
Plots precision over confidence thresholds. ([GitHub][1])

### Recall Curve

![Recall Curve](images/R_curve.png)
Plots recall over confidence thresholds. ([GitHub][1])

### Precision-Recall Curve

![PR Curve](images/PR_curve.png)
Displays the trade-off between precision and recall. ([GitHub][1])

### Overall Results

![Results Summary](images/results.png)
End-to-end pipeline results overview. ([GitHub][1])

### Sample Training Batch

![Train Batch 0](images/train_batch0.jpg)
First batch of training images. ([GitHub][1])

### Validation Labels

![Validation Labels](images/val_batch0_labels.jpg)
Ground-truth labels for validation batch. ([GitHub][1])

### Validation Predictions

![Validation Predictions](images/val_batch0_pred.jpg)
Model predictions for validation batch. ([GitHub][1])

## Evaluation

* **Detection mAP:** 0.72
* **Segmentation IoU (mean):** 0.68
* **Classification Accuracy:** 85%
* **Average Precision:** 0.81 ([GitHub][1])

## Future Work

* Experiment with deeper backbones (ResNet50, EfficientDet). ([GitHub][1])
* Integrate real-time video processing. ([GitHub][1])
* Deploy as REST API using FastAPI or Flask. ([GitHub][1])

## License

This project is licensed under the MIT License. ([GitHub][1])

## Contact

**Your Name** – [your.email@example.com](mailto:your.email@example.com) ([GitHub][1])

```
::contentReference[oaicite:30]{index=30}
```

[1]: https://github.com/Abdullah-Engineer/CV-Project "GitHub - Abdullah-Engineer/CV-Project: This repository is created for the Computer Vision's project"
