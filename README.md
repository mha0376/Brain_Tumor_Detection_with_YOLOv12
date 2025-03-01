
![banner](https://github.com/user-attachments/assets/979b8ccd-3ced-4968-84e7-d5e5ae680fca)


# Brain Tumor Detection Using YOLOv12

This project implements an automated brain tumor detection system leveraging the YOLOv12 object detection model. Trained on a dataset of brain MRI scans, the model identifies and localizes tumors with high accuracy, offering a powerful tool for medical image analysis.

## Table of Contents

*   Project Overview
*   Dataset
*   Installation
*   Usage
    *   Training the Model
    *   Inference
    *   Evaluation
*   Results
*   Contributing

## Project Overview
The objective of this project is to harness the capabilities of YOLOv12, a state-of-the-art object detection framework, to detect brain tumors in MRI images. By fine-tuning a pre-trained YOLOv12 model on a specialized dataset, this project aims to assist in early tumor detection, potentially improving diagnostic efficiency in medical settings.

## Dataset
The dataset utilized is the "[BrainTumor-Br35H](https://universe.roboflow.com/br34h-dataset-brain-tumor/braintumor-br35h)" dataset from Roboflow, accessible via the Roboflow platform (version 3). It consists of annotated brain MRI images categorized into:

*   Training Set: Used to train the YOLOv12 model.
*   Validation set: Used to tune hyperparameters and prevent overfitting. Also, these images can be used due to the lack of test images.

The dataset is formatted for YOLO compatibility, with images and corresponding label files organized accordingly.


## Installation
Follow these steps to set up the project environment:

1.  **Clone the Repository:**
```bash
git clone https://github.com/yourusername/brain-tumor-detection-yolov12.git
cd brain-tumor-detection-yolov12
```
2.  **Install Dependencies:**
Ensure Python 3.8+ is installed, then install the required packages:
```bash
pip install -r requirements.txt
```
Create a requirements.txt file with the following:
```bash
ultralytics
roboflow
supervision
torch
pillow
```
3.  **Download the Dataset:**
Use the Roboflow API to download the dataset:
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="your-api-key-here")
project = rf.workspace("br34h-dataset-brain-tumor").project("braintumor-br35h")
dataset = project.version(3).download("yolov12")
```
Replace "your-api-key-here" with your actual Roboflow API key.

## Usage
### Training the Model
To train the YOLOv12 model on the brain tumor dataset:
1.  **Prepare the Dataset:**
Ensure the dataset is downloaded and structured as per YOLO requirements (e.g., data.yaml file specifying paths to train, validation, and test sets).
2.  **Run the Training Script:**
```bash
# Train the YOLO model on the brain tumor dataset
results = model.train(data="/content/BrainTumor-Br35H-3/data.yaml", epochs=20, imgsz=139, device=device, lr0=0.001)
```
*   Adjust --device to cpu if GPU is unavailable.
*   Modify hyperparameters (e.g., epochs, image size) as needed.

The trained model weights will be saved in runs/detect/train/weights/best.pt.

### Inference
To detect tumors in new MRI images:
1.  **Load the Trained Model:**
```bash
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
```
2.  **Perform Inference:**
```bash
results = model("path/to/image.jpg")
results[0].show()  # Display the image with detections
```
Results are saved in runs/detect/predict/ by default.

### Evaluation
Evaluate the model’s performance using validation data:
1.  **Compute Mean Average Precision (mAP):**
```bash
import supervision as sv
from ultralytics import YOLO

# Load validation dataset
ds = sv.DetectionDataset.from_yolo(
    images_directory_path="/path/to/BrainTumor-Br35H-3/valid/images",
    annotations_directory_path="/path/to/BrainTumor-Br35H-3/valid/labels",
    data_yaml_path="/path/to/BrainTumor-Br35H-3/data.yaml"
)

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Generate predictions and compute mAP
predictions, targets = [], []
for _, image, target in ds:
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    predictions.append(detections)
    targets.append(target)

map_metric = sv.MeanAveragePrecision().update(predictions, targets).compute()
print(f"mAP 50:95: {map_metric.map50_95}")
print(f"mAP 75: {map_metric.map75}")
print(f"mAP 50: {map_metric.map50}")
```
2.  **Visualize Results: Training plots (e.g., confusion matrix, precision-recall curves) are available in runs/detect/train/.**
```bash
confusion_matrix_path = '/content/runs/detect/train/confusion_matrix.png'
confusion_matrix= Image.open(confusion_matrix_path)
max_size = (700, 700)
confusion_matrix.thumbnail(max_size)

display(confusion_matrix)
```

## Results
The fine-tuned YOLOv12 model demonstrates robust performance in brain tumor detection. Key metrics include:
*   mAP 50:95 : 0.690
*   mAP 75 : 0.865
*   mAP 50 : 0.909

Sample outputs and evaluation plots (e.g., confusion matrices, F1 curves) are stored in runs/detect/train/. Before training, the pre-trained model struggles to detect tumors accurately, while post-training inference shows significant improvement.


## Contributing
We welcome contributions to enhance this project. To contribute:
1.  **Fork the repository.**
2.  **Create a feature branch (git checkout -b feature-branch).**
3.  **Commit your changes (git commit -am 'Add new feature').**
4.  **Push to the branch (git push origin feature-branch).**
5.  **Submit a Pull Request.**
