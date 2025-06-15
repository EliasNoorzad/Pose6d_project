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

We trained our YOLO v8 model with the following hyperparameters: the training ran for 23 epochs, and images were resized to a resolution of 640 × 640. The model was initialized with the YOLOv8-nano variant (yolov8n.pt) to balance accuracy and computational efficiency. Training utilized a GPU device (CUDA) to accelerate the process. The configuration and dataset paths were specified in the dataset.yaml file, which clearly defined our training, validation, and testing splits and listed the class labels.

## Results

The trained YOLO model demonstrated strong performance on unseen test data, achieving an overall precision (Box P) of 99.6%, recall of 99.1%, mAP50 of 99.2%, and mAP50-95 of 91.3%. These results closely align with training-set metrics (98.0% precision, 98.3% recall, mAP50 99.1%, and mAP50-95 91.5%), indicating robust generalization capability. Notably, performance across most object classes was consistently high, with the majority surpassing 90% in mAP50-95. However, the "squirrel" class exhibited slightly lower recall (89.7%) and mAP50-95 (69.4%), highlighting potential challenges arising from limited data quality or variability. Overall, the model demonstrates excellent detection and localization performance on novel data, with minor opportunities for improvement in specific object classes.

See detailed implementation steps in [models/YOLO_training.ipynb](models/YOLO_training.ipynb).

## Confusion Matrix Analysis 

The confusion matrix provides a visual summary of the model’s classification performance across all 13 object classes. Most objects such as ape, can, cat, and driller show strong diagonal dominance, indicating high accuracy in correctly predicting their respective classes. However, there are a few off-diagonal values suggesting occasional misclassifications. For example, some instances of squirrel (object 02) were misclassified as background or other objects, which may reflect visual similarities or lower feature distinctiveness. The presence of predictions under the background row implies that the model sometimes fails to associate certain true objects with any known class, possibly due to occlusion or insufficient confidence. Overall, the matrix confirms the model’s high performance, with most predictions aligning correctly.

![confusion_matrix](https://github.com/user-attachments/assets/2bb19b02-def5-473e-847a-cfa60bb4ef3b)

## Recall-Confidence Curve Analysis

Recall is equally strong overall: the network retrieves 99 % of all ground-truth boxes ( R ≈ 0.99 ). Twelve of the thirteen classes achieve essentially perfect recall (1.00 or ≥ 0.998), meaning that almost every instance of ape, camera, pitcher, cat, driller, duck, eggbox, glue, hole-puncher, iron, lamp, and phone is detected. The only clear shortfall is the squirrel (object 02) class at 0.87, indicating that roughly one in eight squirrel instances is still missed—likely because this object appears smaller and with less texture in many frames. Apart from that single outlier, the detector shows excellent coverage, confirming that the original LineMOD train/val splits (without any synthetic augmentation) were sufficient for the model to generalise across viewpoint and scale variations for nearly every object.

![R_curve](https://github.com/user-attachments/assets/515b30ed-729e-44d3-aa9c-757244fa6afb)

## Precision-Confidence Curve Analysis

Precision is consistently high across almost the entire label set: averaged over all 13 objects the detector delivers a 0.994 precision, and eight classes (ape, camera, pitcher, cat, duck, hole-puncher, iron and phone) sit at 0.992 – 0.999, indicating virtually no false positives. Squirrel (0.989) and driller (0.989) trail only slightly, while glue is the lone outlier at 0.971, a value that still reflects good discrimination but suggests occasional confusion with visually similar regions. The perfect 1.000 score for eggbox confirms that every prediction for that class during validation was correct. In short, the model is highly precise overall, with only the glue class showing room for further refinement—perhaps through additional examples or harder negative mining for that object.

![P_curve](https://github.com/user-attachments/assets/aee264a5-493d-4917-b264-da543152d0a1)



