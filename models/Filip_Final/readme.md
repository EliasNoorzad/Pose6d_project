# 6D Object Pose Estimation

This project explores simplified deep learning architectures for **6D object pose estimation**, taking conceptual inspiration from well-known papers in the field, and implements a direct regression approach.

This folder contains two training notebooks for **6D object pose estimation** using the **LineMOD** dataset:

1. **`FinalFilip_4TrainingPoseNet_RGB.ipynb`**  
   - trains a **PoseNet6D** using **RGB images only**.

2. **`FinalFilip_5ExtendingPoseNet_RGB-D.ipynb`**  
   - trains a **PoseNet_RGBD** using **RGB + depth images**.

Both models aim to predict the **6D pose** (3D translation + 3D rotation) of objects in an image.
Both models are evaluated using the **ADD metric**.

---

## Model architecture

The models are inspired by the literature on 6D object pose estimation, particularly **DenseFusion** [1] and **PoseCNN** [2]. 

Our implementation is a **highly simplified PoseNet-style model** [3], based on direct regression of:

- 3D translation vector **t**
- 3D rotation represented as a **quaternion q**.

The extended **PoseNet_RGBD** variant takes inspiration from **DenseFusion** [1], in the sense that it incorporates both RGB and depth information, but without implementing the full DenseFusion pipeline (no dense feature fusion, no iterative refinement).

The model does **not** implement the full PoseCNN architecture:

- No segmentation branch
- No object coordinate regression
- No PnP post-processing.

Both models are therefore best described as **PoseNet-inspired regression networks**, with influence from the PoseCNN and DenseFusion papers.

## Dataset

- Both notebooks use a **custom `PoseEstimationDataset` class**, built on the **LineMOD** dataset.
- The dataset provides:
  - Cropped RGB images.
  - Cropped depth images (optional).
  - Camera intrinsics.
  - Ground truth 6D poses (R, t).
  - 3D object models (used for evaluation with ADD).

---

## Training

- Optimizer: **Adam** with weight decay.
- Learning rate scheduler: **ReduceLROnPlateau**.
- Early stopping based on **ADD metric**.
- Training logs and model checkpoints are saved:
  - Locally (by default).
  - Optionally to Google Drive (if configured).
- Ability to assign custom weights to different translation dimensions (X, Y, Z) in the MSE loss function.

---

## Evaluation

- Models are evaluated using the **ADD metric**:
  - *Average Distance of Model Points*.

- The notebooks provide the possibility to visualize results by:
  - Project model overlay on image (Ground Truth vs Prediction).
  - 3D axes visualization (optional).
  - Plot both full-frame and cropped views.

---

## Results

| Model        | Input                                     | Notes                         |
|--------------|-------------------------------------------|-------------------------------|
| PoseNet      | only cropped RGB image                    | Baseline PoseNet6D model      |
| PoseNet_RGBD | cropped RGB image + cropped depth image   | Extended PoseNet6D with depth |

The performance is visualized during training by, after each finished epoch, printing:

- The MSE loss of the same test image to help interpret performance.
- The average ADD metric for the test set.
- Partwise plotting the translational loss (X, Y, Z) and angular error.

---

## Usage

### 1: PoseNet (RGB only) training

```python
# Run FinalFilip_4TrainingPoseNet_RGB.ipynb
```

### 2: PoseNet_RGBD (RGB + Depth) training

```python
# Run FinalFilip_5ExtendingPoseNet_RGB-D.ipynb
```

---

## Requirements

Install the following python-libraries:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- tqdm

---

## References

[1] Wang, C., Xu, D., Zhu, Y., Martín-Martín, R., Lu, C., Fei-Fei, L., & Savarese, S. (2019).  
DenseFusion: 6D object pose estimation by iterative dense fusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 3343–3352).  
[https://arxiv.org/abs/1901.04780](https://arxiv.org/abs/1901.04780)

[2] Xiang, Y., Schmidt, T., Narayanan, V., & Fox, D. (2017).  
PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes.  
[https://arxiv.org/abs/1711.00199](https://arxiv.org/abs/1711.00199)

[3] Kendall, A., Grimes, M., & Cipolla, R. (2015).  
PoseNet: A convolutional network for real-time 6-DOF camera relocalization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.  
[https://arxiv.org/abs/1505.07427](https://arxiv.org/abs/1505.07427)
