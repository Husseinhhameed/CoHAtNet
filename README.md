# CoHAtNet: An Integrated Convolutional-Transformer Architecture with Hybrid Self-Attention for End-to-End Camera Localization

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.imavis.2025.105674-blue)](https://doi.org/10.1016/j.imavis.2025.105674)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
[![Dataset: 7Scenes](https://img.shields.io/badge/Dataset-7Scenes-orange)](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
[![Dataset: Cambridge Landmarks](https://img.shields.io/badge/Dataset-Cambridge%20Landmarks-orange)](https://www.repository.cam.ac.uk/items/082e0d5e-c893-4d5d-bb0b-1e3c7e8fa6f6)

---

## 📖 Overview
Camera localization estimates the **6DoF (degrees of freedom) pose** of a camera from a single image — a critical task in robotics, autonomous driving, and augmented reality.  
**CoHAtNet** is a novel hybrid convolution-transformer architecture designed for end-to-end camera localization using **RGB** and **RGB-D** data.

Building upon the [CoAtNet](https://github.com/chinhsuanwu/coatnet-pytorch) framework, CoHAtNet integrates **convolutional layers** and **transformer-based attention mechanisms** to capture both local details and global contextual relationships, achieving strong performance in both **small-scale indoor** and **large-scale outdoor** environments.

---

## ✨ Key Features
- **Hybrid Transformer Block** – Combines MBConv layers with transformer-based attention to fuse local and global features.
- **Multi-Modal Support** – Accepts both RGB and RGB-D inputs for enhanced accuracy when depth data is available.
- **End-to-End Localization** – Directly regresses 3D translation and 4D quaternion rotation from images.

---

## 🏗 Architecture
CoHAtNet follows a **five-stage architecture**:

1. **Conv Stem** – Two initial 3×3 convolution layers for low-level feature extraction.
2. **MBConv Blocks** – Capture local spatial features.
3. **Hybrid Transformers** – Fuse convolutional outputs with self-attention for global reasoning.
4. **Global Pooling** – Aggregate spatial features into a single vector.
5. **Pose Regression Head** – Outputs a 7D vector (3D translation + 4D quaternion rotation).

![CoHAtNet Architecture](https://github.com/Husseinhhameed/CoHAtNet/blob/main/CoHAtNet.png)

---

## 📂 Repository Structure
This repository is organized by dataset:

- **7Scenes/** – Scripts for training/testing on the 7Scenes dataset (**RGB-D**).
- **Cambridge_Landmarks/** – Scripts for training/testing on the Cambridge Landmarks dataset (**RGB**).

Each folder includes:
- Dataset preparation scripts
- Training scripts
- Validation scripts
- Testing scripts

---

## 📊 Datasets
CoHAtNet is evaluated on:
- **[7-Scenes Dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)** – Indoor RGB-D dataset.
- **[Cambridge Landmarks Dataset](https://www.repository.cam.ac.uk/items/082e0d5e-c893-4d5d-bb0b-1e3c7e8fa6f6)** – Outdoor RGB dataset.

---

## 📈 Performance

### **7-Scenes Dataset** (Mean Translation Error / Orientation Error)

| Method           | Chess         | Fire          | Heads         | Office         | Pumpkin        | Kitchen        | Stairs         | Avg           |
|------------------|--------------|--------------|--------------|---------------|---------------|---------------|---------------|--------------|
| CoHAtNet-4 (RGB) | 2cm / 0.55°  | 2cm / 0.58°  | 2cm / 1.37°  | 1cm / 0.71°   | 1cm / 0.63°   | 2cm / 0.61°   | 2cm / 0.74°   | 2cm / 0.74°  |
| CoHAtNet-4 (RGB) | 1cm / 0.49°  | 1cm / 0.51°  | 2cm / 1.02°  | 1cm / 0.56°   | 1cm / 0.49°   | 2cm / 0.52°   | 1cm / 0.46°   | 1cm / 0.57°  |

### **Cambridge Landmarks Dataset** (Mean Translation Error / Orientation Error)

| Method           | King’s College | Old Hospital | Shop Facade  | Church        | Avg           |
|------------------|---------------|--------------|--------------|--------------|--------------|
| CoHAtNet-4 (RGB) | 31cm / 0.48°  | 45cm / 0.67° | 16cm / 0.43° | 31cm / 0.70° | 30cm / 0.57° |

---

## 🔧 Loss Function
We adopt the [homography-based loss function](https://github.com/clementinboittiaux/homography-loss-function/blob/main/utils.py), providing stable and interpretable training for camera pose regression.

---

## 🤝 Contributing
Contributions are welcome!  
- Open an **issue** for bug reports or feature requests.  
- Submit a **pull request** for improvements or new features.

📧 For questions, contact: **Hossein Hasan** (hossein.h.hasan@gmail.com)

---

## 📚 Citation
If you use our research in your work, please cite:

Hussein Hasan, Miguel Angel Garcia, Hatem Rashwan, Domenec Puig,  
**CoHAtNet: An integrated convolutional-transformer architecture with hybrid self-attention for end-to-end camera localization**,  
*Image and Vision Computing*, Volume 162, 2025, 105674,  
ISSN 0262-8856, [https://doi.org/10.1016/j.imavis.2025.105674](https://doi.org/10.1016/j.imavis.2025.105674)  
[ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S0262885625002628)  

**Keywords:** Camera localization; Hybrid CNN-transformers; CoAtNet; Hybrid self-attention

```bibtex
@article{hasan2025cohatnet,
  title={CoHAtNet: An integrated convolutional-transformer architecture with hybrid self-attention for end-to-end camera localization},
  author={Hasan, Hussein and Garcia, Miguel Angel and Rashwan, Hatem and Puig, Domenec},
  journal={Image and Vision Computing},
  volume={162},
  pages={105674},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.imavis.2025.105674}
}

