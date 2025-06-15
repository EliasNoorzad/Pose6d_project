# Training with YOLO
We adopted YOLOv8 as our real‐time 2D detector by first converting each object’s RGB images and annotations into the YOLO format. We then trained a single multi‐class YOLOv8 model to recognize all available LineMod objects, enabling it to draw precise bounding boxes and assign correct class labels in one pass. These detections are used directly as inputs—both crops and object IDs—for our downstream Posenet and DenseFusion pose estimation stages.

## Original LineMOD layout

The original dataset was structured per object, containing individual folders named numerically from 01 to 15. Within each object-specific folder, three modalities were provided: RGB color images (rgb/), depth maps (depth/), and segmentation masks (mask/). Each image file was sequentially numbered, such as 0000.png, 0001.png, and so forth. Additionally, a file named gt.yml provided pose information and 2D bounding boxes for each frame. The dataset also included two separate text files, train.txt and test.txt, listing the specific image IDs designated for training and validation respectively.

![original dataset](https://github.com/user-attachments/assets/6acd27ee-b367-4524-8f13-e5addf792fc2)

## YOLO-ready layout

After conversion, the dataset is organized specifically for YOLO training into three distinct subsets: training, validation, and testing, each represented by the top-level folders train/, val/, and test/. Each subset contains an images/ directory holding RGB frames, and a parallel labels/ directory with corresponding .txt files specifying bounding boxes in YOLO-compatible format. File naming convention includes an object-ID prefix (e.g., 01_0004.png and 01_0004.txt), ensuring that images of multiple objects coexist without naming conflicts. The train/ split is used directly for model training, the val/ split serves to tune hyperparameters and prevent overfitting, while the test/ split is reserved strictly for evaluating the final model performance on unseen data.

![yolo](https://github.com/user-attachments/assets/669ddc5b-58bb-44aa-9ab1-47d47d729ad2)

See detailed conversion steps in [dataset/conversion_steps.ipynb](dataset/yolo_conversion_steps.ipynb).

## Training Phase Overview

The training phase involves teaching the YOLO v8 model to accurately identify and localize objects within images. In this step, the model learns visual characteristics of each object category from labeled examples, progressively adjusting its internal parameters to minimize detection errors. We prepared our dataset in a structured format compatible with YOLO's requirements and configured essential parameters—such as dataset paths, class labels, and input dimensions—via a dedicated configuration file (dataset.yaml). Once set up, YOLO systematically processes images through multiple epochs to refine its predictions, ultimately improving detection accuracy.

Below is the configuration file (dataset.yaml) we created, clearly defining paths to our training and validation datasets, and specifying the object classes the YOLO v8 model will learn. This concise setup helps YOLO to seamlessly access the images and corresponding annotations during training and evaluation.

![dataset](https://github.com/user-attachments/assets/ff1a067e-254b-4191-8ca8-832da6388ebf)

## Hyperparameters

We trained our YOLO v8 model with the following hyperparameters: the training ran for 30 epochs, and images were resized to a resolution of 640 × 640. The model was initialized with the YOLOv8-nano variant (yolov8n.pt) to balance accuracy and computational efficiency. Training utilized a GPU device (CUDA) to accelerate the process. The configuration and dataset paths were specified in the dataset.yaml file, which clearly defined our training, validation, and testing splits and listed the class labels.

## Results

The trained YOLO model demonstrated strong performance on unseen test data, achieving an overall precision (Box P) of 99.6%, recall of 99.1%, mAP50 of 99.2%, and mAP50-95 of 91.3%. These results closely align with training-set metrics (98.0% precision, 98.3% recall, mAP50 99.1%, and mAP50-95 91.5%), indicating robust generalization capability. Notably, performance across most object classes was consistently high, with the majority surpassing 90% in mAP50-95. However, the "squirrel" (Object 02) class exhibited slightly lower recall (89.7%) and mAP50-95 (69.4%), highlighting potential challenges arising from limited data quality or variability. Overall, the model demonstrates excellent detection and localization performance on novel data, with minor opportunities for improvement in specific object classes.

See detailed implementation steps in [models/YOLO_training.ipynb](models/YOLO_training.ipynb).

## Confusion Matrix Analysis 

The confusion matrix obtained from the test set confirms strong classification performance for nearly all object classes, as indicated by the dominant diagonal entries representing correct detections. Most classes—such as "ape," "camera," "pitcher," "cat," and others—achieved perfect accuracy, with all 100 ground-truth instances correctly identified. However, the "squirrel" (Object 02) class showed some difficulty, with 89 out of 96 instances correctly detected, while 7 false positives and 8 missed detections appeared, contributing to its lower recall. The "glue", "phone" classes also experienced minor detection errors. Overall, the matrix indicates excellent generalization capability, highlighting only slight areas for further refinement, particularly for the "squirrel" class.

![confusion_matrix (1)](https://github.com/user-attachments/assets/1782f6dc-907f-43f2-8800-17749a339490)

## Recall-Confidence Curve Analysis

The evaluation on the test set confirms strong overall performance, with a recall of 99.1% and mAP50-95 of 91.3%, demonstrating robust generalization to unseen data. Analysis of the recall-confidence curve reveals excellent recall across almost all object classes, consistently achieving near-perfect detection even at relatively high confidence thresholds. However, performance for the "squirrel" (Object 02) class remains comparatively weaker, with a recall of 89.7% and mAP50-95 of 69.4%. The curve indicates this class suffers from systematic detection misses, likely due to data-related issues rather than model limitations. The "glue" class also shows slightly lower localization quality (mAP50-95 of 87.8%) and minor recall drops at higher confidence thresholds. Apart from these two cases, all other classes maintain very high precision, recall, and localization quality, highlighting the model's effectiveness across the majority of object categories. 

![R_curve (1)](https://github.com/user-attachments/assets/a7a2034a-3ea5-4b6c-92bd-f5286771e5a8)

## Precision-Confidence Curve Analysis

Precision is consistently high across almost the entire label set: averaged over all 13 objects the detector delivers a 0.994 precision, and eight classes (ape, camera, pitcher, cat, duck, hole-puncher, iron and phone) sit at 0.992 – 0.999, indicating virtually no false positives. Squirrel (0.989) and driller (0.989) trail only slightly, while glue is the lone outlier at 0.971, a value that still reflects good discrimination but suggests occasional confusion with visually similar regions. The perfect 1.000 score for eggbox confirms that every prediction for that class during validation was correct. In short, the model is highly precise overall, with only the glue class showing room for further refinement—perhaps through additional examples or harder negative mining for that object.

![P_curve](https://github.com/user-attachments/assets/aee264a5-493d-4917-b264-da543152d0a1)



