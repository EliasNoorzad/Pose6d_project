# Training with YOLO (Object Detection)
We adopted YOLOv8 as our real‐time 2D detector by first converting each object’s RGB images and annotations into the YOLO format. We then trained a single multi‐class YOLOv8 model to recognize all available LineMod objects, enabling it to draw precise bounding boxes and assign correct class labels in one pass. These detections are used directly as inputs—both crops and object IDs—for our downstream Posenet and DenseFusion pose estimation stages.

## Original LineMOD layout

The original dataset was organized per object (01–15), each containing RGB images, depth maps, segmentation masks, and a gt.yml file with pose and 2D bounding box data. Image IDs for training and validation were listed in train.txt and test.txt.

After conversion, the dataset was restructured into train/, val/, and test/ folders, each with images/ and labels/ subdirectories. Labels follow YOLO format with normalized bounding boxes and class labels. Filenames include object IDs (e.g., 01_0004.png) to avoid conflicts.

See detailed conversion steps in [dataset/conversion_steps.ipynb](dataset/yolo_conversion_steps.ipynb).

## Training Phase Overview

The YOLOv8 model was trained to detect and localize objects by learning from labeled examples. A structured dataset was prepared, and key parameters such as dataset paths, class labels, and input size were defined in a dataset.yaml configuration file. This setup allowed YOLO to efficiently process the data over multiple epochs, gradually improving its detection accuracy.

![dataset](https://github.com/user-attachments/assets/ff1a067e-254b-4191-8ca8-832da6388ebf)

## Training YOLO

We trained our YOLO v8 model with the following hyperparameters: the training ran for 30 epochs, and images were resized to a resolution of 640 × 640. The model was initialized with the YOLOv8-nano variant (yolov8n.pt) to balance accuracy and computational efficiency. Training utilized a GPU device (CUDA) to accelerate the process. The configuration and dataset paths were specified in the dataset.yaml file, which clearly defined our training, validation, and testing splits and listed the class labels.

See detailed implementation steps in [models/YOLO_training.ipynb](models/YOLO_training.ipynb).

# 6D Object Pose Estimation

This project explores simplified deep learning architectures for **6D object pose estimation**, taking conceptual inspiration from well-known papers in the field, and implements a direct regression approach.
There are two training notebooks provided for 6D object pose estimation using the LineMOD dataset:

 [models/training_RGB.ipynb](models/training_RGB.ipynb) trains a PoseNet6D model using RGB images only.

 [models/training_RGBD.ipynb](models/training_RGBD.ipynb) trains a PoseNet6D_RGBD model using both RGB and depth images.

Each notebook includes all necessary preprocessing steps, from loading and formatting the data to preparing it for model input. Both models are designed to predict the 6D pose of objects, including 3D translation and rotation, and are evaluated using the ADD metric.

**Note:** The trained model weights (.pt files) were not included in the repository due to GitHub’s file size limitations, which restrict individual file uploads to a maximum of 25 MB. To access the weights, an external download link can be provided upon request or hosted through a suitable file-sharing service.

## Model architecture

The models are inspired by the literature on 6D object pose estimation, particularly **DenseFusion**  and **PoseCNN**.

Our implementation is a **highly simplified PoseNet-style model** , based on direct regression of:

- 3D translation vector **t**
- 3D rotation represented as a **quaternion q**.

The extended **PoseNet_RGBD** variant takes inspiration from **DenseFusion**, in the sense that it incorporates both RGB and depth information, but without implementing the full DenseFusion pipeline (no iterative refinement for example).

The model does **not** implement the full PoseCNN architecture:

- No segmentation branch
- No object coordinate regression
- No PnP post-processing.

The models are therefore best described as **PoseNet-inspired regression networks**, with the RGB-D extension being influenced by the PoseCNN and DenseFusion papers.

Access the complete report here: [Report.pdf](Report.pdf)

# How to Run and Test the 6D Object Pose Estimation Pipeline (YOLO + PoseNet)
The provided scripts allow you to run and test a complete 6D object pose estimation pipeline.

1. **Object Detection** using a YOLOv8 model.
2. **Pose Estimation** using two different PoseNet6D variants:
   - [models/tryModel_RGB.ipynb](models/tryModel_RGB.ipynb): tests RGB-only model.
   - [models/tryModel_RGBD.ipynb](models/tryModel_RGBD.ipynb): tests RGB-D model that uses both RGB and depth information.

## Pipeline Overview

Each notebook follows this sequence:

1. **Load YOLOv8 model** and perform object detection on RGB images.
2. **Crop detected object regions** from RGB image (and optionally from depth map).
3. **Prepare inputs** (e.g., camera intrinsics, normalized crops).
4. **Run PoseNet6D** (RGB or RGBD) to predict the 6D object pose:
   - 3D translation vector.
   - 3D rotation (quaternion).
5. **Visualize predictions** alongside ground-truth poses.
6. **Evaluate performance** using the ADD metric.

## Requirements

Install the required Python packages using:

```bash
pip install torch torchvision \
            opencv-python \
            numpy \
            matplotlib \
            tqdm \
            ultralytics \
            scipy \
            pillow \
            scikit-image
```

---

## How to Run

### Option 1: Local Jupyter

```bash
jupyter notebook tryModel_RGB.ipynb
# or
jupyter notebook tryModel_RGBD.ipynb
```

### Option 2: Google Colab

You can also upload the notebooks and model files to **Google Colab**.

> **Important:** You must upload or mount your model checkpoint files and dataset manually.
>
> - Upload the pretrained YOLOv8 model (called 'yolo.pt') together with the PoseNet6D weights (`posenet6d_RGB.pt`, `posenet6d_RGBD.pt`).
> - Make sure paths inside the notebook are updated accordingly to match the runtime environment (ex. `/content/` in Colab).
> - Use `drive.mount()` in Colab to access your files.

---

## Required Files

- YOLOv8 model weights (`yolo.pt`).
- PoseNet RGB model (`posenet6d_RGB.pt`).
- PoseNet RGBD model (`posenet6d_RGBD.pt`)
- Dataset in LineMOD format with aligned RGB, depth, and ground truth poses.
- Camera intrinsics for projection.

---

## Output

- Predicted 6D poses (translation + rotation).
- Visual comparisons between predictions and ground-truth.
- ADD metric (average distance of model points).
