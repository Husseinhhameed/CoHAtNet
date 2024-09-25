# CoHAtNet: An Integrated Convolution-Transformer Architecture for End-to-End Camera Localization

## Overview
Camera localization is the process of estimating the 6DoF (degrees of freedom) pose of a camera from a given image, playing a crucial role in robotics, autonomous driving, and augmented reality. CoHAtNet is a novel hybrid convolution-transformer architecture designed for end-to-end camera localization using RGB and RGB-D data. 

CoHAtNet builds on the CoAtNet framework by integrating convolutional layers and attention mechanisms to capture both local details and global contextual relationships from input images. It is designed to perform effectively across both small-scale indoor and large-scale outdoor environments.

## Key Features
- **Hybrid Transformer Block:** CoHAtNet combines Mobile Inverted Bottleneck Convolution (MBConv) layers with transformer-based attention to form hybrid transformers, enabling better performance by fusing local and global features.
- **Support for RGB and RGB-D Data:** The model can take RGB or RGB-D images as input, improving accuracy in environments where depth information is available.
- **End-to-End Localization:** Trained for camera localization tasks directly from images, estimating the camera’s translation and orientation.

## Architecture
CoHAtNet has a five-stage architecture:
1. **Conv Stem:** Initial stage with two 3x3 convolution layers for feature extraction.
2. **MBConv Blocks:** Layers to capture local features and interactions.
3. **Hybrid Transformers:** Attention mechanisms enriched by convolution outputs to capture global and local dependencies.
4. **Global Pooling:** Final stage for aggregating global information before pose regression.
5. **Pose Regression Head:** Outputs the camera’s 3D translation and orientation (quaternion) as 7 values.

![CoHAtNet Architecture](https://github.com/Husseinhhameed/CoHAtNet/blob/main/CoHAtNet.png)



## Repository Structure
In this repository, we have separate folders for each dataset:
- **7Scenes Dataset Folder:** Contains all the necessary scripts for training the CoHAtNet model on the 7Scenes dataset (RGB-D data).
- **Cambridge Landmarks Dataset Folder:** Includes the scripts for training on the Cambridge Landmarks dataset (RGB data).

Each folder contains the scripts for Dataset preparing,training, validation, and testing the model. You can run the scripts directly or modify them based on your custom dataset.

In addition, we provide a complete implementation of CoHAtNet for both datasets in an **IPython Notebook (.ipynb)** file, which can be executed directly on Google Colab.

### Google Colab Setup
To run the model on Google Colab:
1. **Upload the Dataset to Google Drive:** The dataset must be stored in your Google Drive account. Ensure that the dataset is organized according to the expected structure (images and ground truth files in the appropriate directories).
2. **Open the Colab Notebook:** The provided `.ipynb` file contains a step-by-step guide for training and evaluating CoHAtNet.
3. **Mount Google Drive:** The notebook includes commands to mount your Google Drive so that the dataset can be accessed directly from Colab.
4. **Run the Notebook:** Execute the cells to load the data, train the model, and evaluate performance.


## Datasets
CoHAtNet is evaluated on two benchmark datasets:
- **7-Scenes Dataset:** Contains indoor scenes with RGB-D images.
- **Cambridge Landmarks Dataset:** A large-scale dataset for outdoor camera localization using RGB images.

## Performance
Experimental results show that CoHAtNet outperforms previous state-of-the-art CNN and transformer-based approaches. The model demonstrates competitive accuracy in both indoor and outdoor environments.
### 7-Scenes Dataset Results (Mean Translation Error / Orientation Error)

| Method                | Chess | Fire | Heads | Office | Pumpkin | Kitchen | Stairs | Avg  |
|-----------------------|-------|------|-------|--------|---------|---------|--------|------|
| CoHAtNet-4 (RGB)      | 2cm / 0.55° | 2cm / 0.58° | 2cm / 1.37° | 1cm / 0.71° | 1cm / 0.63° | 2cm / 0.61° | 2cm / 0.74° | 2cm / 0.74° |

### Cambridge Landmarks Dataset Results (Mean Translation Error / Orientation Error)

| Method                | King’s College | Old Hospital | Shop Facade | Church | Avg  |
|-----------------------|----------------|--------------|-------------|--------|------|
| CoHAtNet-4 (RGB)      | 31cm / 0.48° | 45cm / 0.67° | 16cm / 0.43° | 31cm / 0.70° | 30cm / 0.57° |


## Homography Loss Function
The homography-based loss function is used from the implementation provided [here](https://github.com/clementinboittiaux/homography-loss-function/blob/main/utils.py). This loss function provides a more stable and interpretable approach for camera pose regression tasks.

## Original CoAtNet Implementation
we used the original CoAtNet implementation, on which CoHAtNet is based,  it can be found [here](https://github.com/chinhsuanwu/coatnet-pytorch).
