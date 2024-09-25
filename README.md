# CoHAtNet: An Integrated Convolution-Transformer Architecture for End-to-End Camera Localization

### Authors: 
- Hussein Hasan, Universitat Rovira i Virgili, Spain  
- Miguel Angel Garcia, Autonomous University of Madrid, Spain  
- Hatem Rashwan, Universitat Rovira i Virgili, Spain  
- Domenec Puig, Universitat Rovira i Virgili, Spain  

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

![CoHAtNet Architecture](path_to_your_image)

## Datasets
CoHAtNet is evaluated on two benchmark datasets:
- **7-Scenes Dataset:** Contains indoor scenes with RGB-D images.
- **Cambridge Landmarks Dataset:** A large-scale dataset for outdoor camera localization using RGB images.

## Performance
Experimental results show that CoHAtNet outperforms previous state-of-the-art CNN and transformer-based approaches. The model demonstrates competitive accuracy in both indoor and outdoor environments.

### 7-Scenes Dataset Results (Mean Translation Error / Orientation Error)


