# Brain Tumor Segmentation using Deep Learning

This repository presents a complete pipeline for **brain tumor segmentation** using deep learning and image processing techniques. The project involves training a custom model on medical image data, visualizing results, and evaluating performance with various metrics and graphical analyses.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

Brain tumor detection and segmentation are vital for early diagnosis and treatment. This project uses deep learning techniques to accurately segment tumors from MRI scans. It includes data preprocessing, model training, and evaluation using confusion matrices, PR curves, and batch visualization.

---

## Dataset

The dataset used contains labeled MRI images with masks indicating tumor locations. Data was split into training, validation, and test sets with augmentations applied during training.

The dataset is available at the following link:

[Tumor Detection Dataset](https://universe.roboflow.com/brain-tumor-detection-wsera/tumor-detection-ko5jp/dataset/8)

---

## Project Structure

```

Brain\_Tumor\_Segmentation
├── Brain\_Tumor\_Segmentation.ipynb
├── runs/  # YOLOv5 training outputs
├── images/
│   ├── confusion\_matrix.png
│   ├── confusion\_matrix\_normalized.png
│   ├── 10\_random\_predict.png
│   ├── 10\_first\_predict.png
│   ├── labels.jpg
│   ├── labels\_correlogram.jpg
│   ├── P\_curve.png
│   ├── R\_curve.png
│   ├── PR\_curve.png
│   ├── results.png
│   ├── train\_batch0.jpg
│   ├── val\_batch0\_labels.jpg
│   ├── val\_batch0\_pred.jpg
├── README.md
├── requirements.txt

````

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Abdullah-Engineer/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
````

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the main notebook to execute the pipeline:

```bash
jupyter notebook Brain_Tumor_Segmentation.ipynb
```

Ensure that the dataset is in the correct path and formatted appropriately.

---

## Training Details

* **Model Used:** Custom segmentation model using CNN-based architecture
* **Optimizer:** Adam
* **Loss Function:** Dice Loss + BCE
* **Epochs:** 20+
* **Learning Rate:** 1e-4
* **Data Augmentation:** Horizontal flip, rotation, scaling

---

## Results

* Achieved high segmentation accuracy on validation data
* Visual metrics indicate reliable model performance
* Confusion matrices and performance curves included

### Metrics

* Accuracy
* Precision, Recall
* F1 Score
* Dice Coefficient
* IoU

---

## Evaluation

### Confusion Matrix

| Standard                                                                     | Normalized                                                                                           |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| ![Confusion Matrix](images/confusion_matrix.png "Standard Confusion Matrix") | ![Confusion Matrix Normalized](images/confusion_matrix_normalized.png "Normalized Confusion Matrix") |

### Precision & Recall

* ![Precision Curve](images/P_curve.png "Precision Curve")
* ![Recall Curve](images/R_curve.png "Recall Curve")
* ![PR Curve](images/PR_curve.png "Precision-Recall Curve")

---

## Visualizations

### Predictions vs Ground Truth

\| ![10 Random Predictions](images/10_random_predict.png "10 Random Predictions") | ![First 10 Predictions](images/10_first_predict.png "First 10 Predictions") |

### Label & Correlogram Analysis

* ![Labels](images/labels.jpg "Class Label Distribution")
* ![Correlogram](images/labels_correlogram.jpg "Label Correlogram")

### Batch Previews

\| ![Train Batch](images/train_batch0.jpg "Example Training Batch") | ![Val Labels](images/val_batch0_labels.jpg "Validation Ground Truth Masks") | ![Val Predictions](images/val_batch0_pred.jpg "Validation Predicted Masks") |

### Overall Results

* ![Results Graph](images/results.png "Results Overview Graph")

---

## Acknowledgements

* Dataset from [Tumor Detection Dataset](https://universe.roboflow.com/brain-tumor-detection-wsera/tumor-detection-ko5jp/dataset/8)
* Inspired by medical segmentation papers and YOLO-based architectures
* Libraries: `PyTorch`, `OpenCV`, `Matplotlib`, `Seaborn`, `NumPy`, etc.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
::contentReference[oaicite:6]{index=6}
```

[1]: https://stackoverflow.com/questions/63097837/markdown-image-titles?utm_source=chatgpt.com "Markdown image titles - Stack Overflow"
[2]: https://meta.stackexchange.com/questions/142750/add-image-captions?utm_source=chatgpt.com "Add Image Captions - Meta Stack Exchange"
[3]: https://github.com/jehna/readme-best-practices?utm_source=chatgpt.com "jehna/readme-best-practices - GitHub"
[4]: https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes?utm_source=chatgpt.com "About READMEs - GitHub Docs"
[5]: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables?utm_source=chatgpt.com "Organizing information with tables - GitHub Docs"
[6]: https://stackoverflow.com/questions/39378020/how-to-display-table-in-readme-md-file-in-github?utm_source=chatgpt.com "How to display Table in README.md file in Github? - Stack Overflow"
[7]: https://dev.to/stephencweiss/markdown-image-titles-and-alt-text-5fi1?utm_source=chatgpt.com "Markdown Image Titles and Alt Text - DEV Community"
[8]: https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/?utm_source=chatgpt.com "How to Write a Good README File for Your GitHub Project"
[9]: https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax?utm_source=chatgpt.com "Basic writing and formatting syntax - GitHub Docs"
