# Brain Tumor Segmentation with YOLO 11

This project showcases brain tumor segmentation in MRI images using the YOLO 11 model from Ultralytics. Developed as part of an internship at ARCH TECHNOLOGIES, it aims to accurately detect and segment brain tumors, aiding in medical image analysis for diagnosis and treatment planning.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. **Install Required Packages**:

   Ensure Python 3.8+ is installed. Use a virtual environment for best practice, then install dependencies:

   ```bash
   pip install ultralytics gdown torch matplotlib pillow
   ```

   **Note**: A GPU (e.g., NVIDIA T4) is recommended for faster training. This project was tested on Google Colab.

## Dataset

The dataset consists of MRI images with annotated brain tumors, split into training, validation, and test sets. It’s hosted on Google Drive and can be downloaded as follows:

```bash
gdown --id 1A2ULxzuAKKPYuhwErFxDH__C9SNVdNdm --output dataset.zip
unzip dataset.zip -d Tumor_Dataset
```

The resulting `Tumor_Dataset` directory has this structure:

- `train/`: Training images and labels
- `valid/`: Validation images and labels
- `test/`: Test images and labels
- `data.yaml`: YOLO configuration file

## Usage

The project is implemented in a Jupyter notebook (`Brain_Tumor_Segmentation.ipynb`). Here’s how to use it:

1. **Prepare the Dataset**:

   Download and extract the dataset as described in the [Dataset](#dataset) section.

2. **Run the Notebook**:

   Open `Brain_Tumor_Segmentation.ipynb` in Jupyter Notebook or Google Colab and execute the cells sequentially.

   - **Training**: Trains the YOLO 11 segmentation model.
   - **Prediction**: Generates segmentation masks on test images.

3. **Command-Line Alternative**:

   - **Train the Model**:

     ```python
     from ultralytics import YOLO

     model = YOLO('yolo11n-seg.pt')
     model.train(data='Tumor_Dataset/data.yaml', epochs=100, batch=16, name='yolo11n_seg_custom')
     ```

   - **Predict on Test Images**:

     ```python
     test_images = 'Tumor_Dataset/test/images'
     results = model.predict(source=test_images, save=True)
     ```

   Results are saved in `runs/segment/train` (training) and `runs/segment/predict` (predictions).

## Results

The model was trained for 100 epochs, achieving promising segmentation performance. Below are key visualizations:

- **Confusion Matrix**: Shows classification accuracy across tumor types.

  ![Confusion Matrix](assets/confusion_matrix.png)

- **Precision-Recall Curve**: Illustrates the balance between precision and recall.

  ![PR Curve](assets/PR_curve.png)

- **Sample Predictions**: Segmentation masks on 10 random test images.

  ![10 Random Predictions](assets/10_random_predict.png)

Additional results (e.g., `results.png`, `P_curve.png`, `R_curve.png`) are available in the `runs/segment/train` directory after training, and more prediction images are in `runs/segment/predict`.

## Contributing

Contributions are welcome! Please submit issues or pull requests for enhancements, bug fixes, or documentation updates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ARCH TECHNOLOGIES**: For the internship opportunity.
- **Ultralytics**: For the YOLO 11 framework.
- **Dataset**: Sourced from Google Drive (original provider unspecified).