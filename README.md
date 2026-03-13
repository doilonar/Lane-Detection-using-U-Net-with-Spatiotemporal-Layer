# Lane-Detection-using-U-Net-with-Spatiotemporal


This repository contains implementations for lane detection using both traditional computer vision techniques and deep learning models. The project explores various U-Net architectures, including a version enhanced with ConvLSTM layers to incorporate spatiotemporal information for more robust video-based lane detection.

## Features

*   **Classical Computer Vision Approach**: Lane detection using OpenCV, featuring color space transformation, perspective warping, and polynomial fitting.
*   **U-Net with Binary Focal Loss**: A U-Net model trained for lane segmentation using Binary Focal Loss, which is effective for handling class imbalance.
*   **U-Net with ConvLSTM and IoU Loss**: An advanced U-Net model incorporating `ConvLSTM2D` layers at its bottleneck to leverage temporal data from video sequences. This model is trained using an IoU (Intersection over Union) based loss function.
*   **Training and Inference Scripts**: Includes scripts to train the models from scratch and to run inference on single images and video files.

## Repository Structure

```
.
├── Damoc_Robert-Marian.pptx      # Project presentation
├── Damoc_Robert-Marian_licenta.docx # Project thesis document
├── TuSimple_output_generation.py # Classical computer vision lane detection implementation
├── test_image.py                 # Script to test a trained model on a single image
├── video_unet.py                 # Script to run lane detection on a video using a U-Net model
├── unet_binaryfocal/
│   ├── lane_cv_dropout_batch.h5  # Pre-trained U-Net model (Focal Loss)
│   └── run_unet.py               # Training script for the U-Net with Binary Focal Loss
├── unet_iou_loss/
│   └── lane_cv_dropout_batch.h5  # Pre-trained U-Net model (IoU Loss)
└── unet_lstm/
    └── run_unet.py               # Training script for the U-Net + ConvLSTM model with IoU Loss
```

## Models

### 1. Classical CV Lane Detection (`TuSimple_output_generation.py`)
This method uses a pipeline of traditional image processing techniques:
1.  **Preprocessing**: Converts the image to grayscale and HSV color spaces to create a binary mask for yellow and white lane lines.
2.  **Perspective Transform**: Warps the detected lane lines into a bird's-eye view.
3.  **Lane Fitting**: Uses a sliding window approach on a histogram of the warped image to identify lane pixels and fits a second-degree polynomial to each lane line.
4.  **Visualization**: Unwarps the detected lane area and overlays it on the original frame, displaying the calculated radius of curvature.
5.  **This project utilizes the TuSimple dataset.**

### 2. U-Net with Binary Focal Loss (`unet_binaryfocal/`)
This is a standard U-Net architecture designed for semantic segmentation.
-   **Architecture**: Consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) for precise localization.
-   **Loss Function**: Uses `BinaryFocalLoss` to address the imbalance between the lane pixels and the background.
-   **Training**: The `run_unet.py` script uses a data generator to feed images and corresponding masks to the model.
### Accuracy Binary Focal Loss
![Accuracy](https://github.com/user-attachments/assets/65905bbd-3a0a-463b-9a96-c32a39d4193b)
### Compared Binary Focal Loss
![Compared](https://github.com/user-attachments/assets/19bd1bd0-edb0-4ec7-83f2-4107568f7cc9)
### Loss Binary Focal Loss
![Loss](https://github.com/user-attachments/assets/bb103422-44d2-4d46-a567-f01b18250b58)
### 3. U-Net with ConvLSTM (`unet_lstm/`)
This model enhances the standard U-Net by adding `ConvLSTM2D` layers in the bottleneck. This allows the model to learn spatiotemporal features from sequential frames in a video, improving temporal consistency.
-   **Architecture**: Integrates two `ConvLSTM2D` layers between the encoder and decoder paths.
-   **Loss Function**: Utilizes an `iou_loss` (1 - IoU), which directly optimizes the Intersection over Union metric. This is highly effective for segmentation tasks.
-   **Metric**: The model is evaluated using the `iou` metric.
### Accuracy IoU Loss
![Accuracy IoU Loss](https://github.com/user-attachments/assets/be6ff3a1-e1e1-49f1-b382-4d973209578b)
### Compared IoU Loss
![Compared IoU Loss](https://github.com/user-attachments/assets/82d07aef-f242-482d-b5c2-af95b807e81a)
### Loss IoU Loss
![Loss IoU Loss](https://github.com/user-attachments/assets/009f6fc6-4f2d-4caf-9841-4a6dba9d103c)

### Accuracy LSTM Integration
![Accuracy LSTM Integration](https://github.com/user-attachments/assets/3abaa68e-fefe-4405-a6fd-da09fccf0ec9)

### Loss LSTM Integration
![Loss LSTM Integration](https://github.com/user-attachments/assets/1dbd6bdb-e2b5-492b-85b6-f39be030060f)
## How to Use

### Prerequisites
You need Python 3 and the following libraries:
-   TensorFlow
-   OpenCV
-   NumPy
-   Matplotlib
-   focal-loss

Install them using pip:
```bash
pip install tensorflow opencv-python numpy matplotlib focal-loss
```

### Training a Model
The training scripts (`run_unet.py`) expect the dataset to be organized into two main directories: one for the input images and one for the ground truth masks.

1.  Place your training images in a folder (e.g., `lane/images/`).
2.  Place the corresponding segmentation masks in another folder (e.g., `lane_detect/masks/`).
3.  Update the `image_folder` and `mask_folder` paths inside the desired `run_unet.py` script.
4.  Execute the script to start training:
    ```bash
    # For U-Net with Binary Focal Loss
    python unet_binaryfocal/run_unet.py

    # For U-Net with ConvLSTM and IoU Loss
    python unet_lstm/run_unet.py
    ```
    The trained model will be saved as `lane_cv_dropout_batch.h5` in the same directory.

### Inference on a Single Image
The `test_image.py` script performs lane detection on a single image and displays the input image, the predicted mask, and a filtered/overlayed image.

1.  Open `test_image.py`.
2.  Set the `model` variable by loading the desired pre-trained model file (e.g., `unet_lstm/lane_cv_dropout_batch.h5`).
3.  Update the `image_path` to point to your test image.
4.  Run the script:
    ```bash
    python test_image.py
    ```

### Inference on a Video
The `video_unet.py` script processes a video file frame by frame, overlays the predicted lane segmentation, and displays the result in real-time.

1.  Open `video_unet.py`.
2.  Update `input_video_path` to your video file.
3.  Set the `model` variable by loading the desired pre-trained model. The `unet_lstm` model is recommended for video.
4.  Run the script:
    ```bash
    python video_unet.py
    ```

### Without LSTM Integration
![Without LSTM Integration](https://github.com/user-attachments/assets/17c628cf-bbd6-40b0-adc8-0f9f6c63602f)
### With LSTM Integration
![With LSTM Integration](https://github.com/user-attachments/assets/6cd5526f-cf9e-4a04-a61e-1b8895eb1a99)
### Research Paper
https://ieeexplore.ieee.org/abstract/document/11105626
