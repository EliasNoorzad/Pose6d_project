# Training with YOLO
We adopted YOLOv8 as our real‐time 2D detector by first converting each object’s RGB images and annotations into the YOLO format. We then trained a single multi‐class YOLOv8 model to recognize all available LineMod objects, enabling it to draw precise bounding boxes and assign correct class labels in one pass. These detections are used directly as inputs—both crops and object IDs—for our downstream Posenet and DenseFusion pose estimation stages.

## Original LineMOD layout

Each object (e.g., 01, 02, … 15) lives in its own directory that contains every sensing modality and annotation in one place: an rgb/ folder with all colour frames (0000.png, 0001.png, …), matching depth/ and mask/ folders, the gt.yml file holding per-frame 6-DoF pose plus 2-D bounding boxes, and two plain-text lists (train.txt and test.txt) that define which image IDs belong to training and validation.

![original dataset](https://github.com/user-attachments/assets/6acd27ee-b367-4524-8f13-e5addf792fc2)

## YOLO-ready layout

After conversion, the same data are split by purpose rather than by modality. Two top-level folders — train/ and val/ — each hold an images/ sub-folder with the RGB frames and a labels/ sub-folder with companion .txt files whose four normalised numbers encode the bounding-box centre and size. Every filename was prefixed with the object ID (e.g., 04_0933.png, 04_0933.txt) so that images from different objects coexist without collisions, and a single dataset.yaml lists the 13 class names YOLO sees during training.

![yolo](https://github.com/user-attachments/assets/7b5c87e1-a948-4ab0-8b03-9cc40fcf95aa)

See detailed conversion steps in [dataset/conversion_steps.ipynb](dataset/yolo_conversion_steps.ipynb).

## Training Phase Overview

The training phase involves teaching the YOLO v8 model to accurately identify and localize objects within images. In this step, the model learns visual characteristics of each object category from labeled examples, progressively adjusting its internal parameters to minimize detection errors. We prepared our dataset in a structured format compatible with YOLO's requirements and configured essential parameters—such as dataset paths, class labels, and input dimensions—via a dedicated configuration file (dataset.yaml). Once set up, YOLO systematically processes images through multiple epochs to refine its predictions, ultimately improving detection accuracy.

Below is the configuration file (dataset.yaml) we created, clearly defining paths to our training and validation datasets, and specifying the object classes the YOLO v8 model will learn. This concise setup helps YOLO to seamlessly access the images and corresponding annotations during training and evaluation.

![dataset](https://github.com/user-attachments/assets/d5eaf458-351a-4cd6-8fbd-177151280897)


