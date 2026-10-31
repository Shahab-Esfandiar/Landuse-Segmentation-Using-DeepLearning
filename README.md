
<img width="400" height="400" alt="images" src="https://github.com/user-attachments/assets/2408b3c6-cad2-4072-bebc-7925a4159549" />

# Comparative Benchmark of Deep Convolutional Architectures for High-Resolution Land Cover Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Framework-red.svg)](https://keras.io/)

---

## 1. Study Overview & Objectives

Accurate spatial delineation of thematic land cover classes is fundamental to urban planning, environmental monitoring, and disaster damage assessment. High-resolution orthophotography introduces significant challenges, including high intra-class variance, irregular geometric boundaries, and severe class imbalance.

This project implements an end-to-end deep learning research pipeline to:
* Extract and curate information-dense tiles from large-scale GeoTIFF imagery.
* Mitigate background sparsity and class imbalance through selective filtering and augmentation.
* Benchmark four encoder-decoder topologies under identical optimization and hyperparameter configurations.
* Evaluate segmentation fidelity using statistical reliability metrics: **Overall Accuracy (OA)**, **Cohen's Kappa ($\kappa$)**, and the **Jaccard Index (mIoU)**.

---

## 2. Dataset Taxonomy & Class Schema

The target masks comprise 5 mutually exclusive semantic classes:

| Class ID | Class Name | Mask Color | Description & Geometry |
| :---: | :--- | :---: | :--- |
| **0** | **Background / Unlabeled** | Black | Non-target or unannotated pixels; excluded from loss dominance via filtering. |
| **1** | **Building Footprints** | White | Discrete spatial footprints requiring sharp orthogonal edge preservation. |
| **2** | **Woodland / Vegetation** | Green | Heterogeneous texture and spectral response; irregular contiguous regions. |
| **3** | **Water Bodies** | Blue | High spectral absorption, uniform interior with distinct boundary transitions. |
| **4** | **Road Infrastructure** | Red | Continuous linear network; highly sensitive to spatial disconnections and occlusions. |

---

## 3. Methodology & Workflow Pipeline

The experimental process follows a systematic remote sensing workflow:

```text
Raw GeoTIFF Orthophotos & Target Masks
                 │
                 ▼
[Step 1: Non-Overlapping Tiling]
  └─ Extract uniform 512 × 512 px patches via sliding window
                 │
                 ▼
[Step 2: Information Sparsity Filtering]
  └─ Discard patches where non-background area < 5%
                 │
                 ▼
[Step 3: Geographic Sampling & Partitioning]
  └─ Stratified extraction: 400 Train, 160 Validation, 10 Test
                 │
                 ▼
[Step 4: Spatial Data Augmentation]
  └─ Geometric reflections (Horizontal & Vertical Flips)
                 │
                 ▼
[Step 5: Categorical Encoding]
  └─ One-Hot encoding of ground truth masks into (512, 512, 5) tensors
                 │
                 ▼
[Step 6: Model Training & Optimization]
  └─ Train U-Net, ResNet-34, VGG-16, and Inception-V3 (50 Epochs, Adam)
                 │
                 ▼
[Step 7: Quantitative Evaluation]
  └─ Overall Accuracy, Cohen's Kappa, and Jaccard Index (mIoU)
```

### Preprocessing Specifications
* **Patch Generation:** Images and corresponding raster masks are sectioned into non-overlapping tiles of $512 \times 512 \times 3$.
* **Zero-Information Elimination:** Tiled patches with background coverage exceeding $95\%$ are removed ($\sum_{c=1}^4 y_c / \sum_{c=0}^4 y_c \ge 0.05$) to avoid uninformative gradient updates.
* **Balanced Sampling:** A subset of 10 patches per geographic scene is extracted, yielding **400 training**, **160 validation**, and **10 unseen test patches**.
* **Data Augmentation:** Horizontal (`np.fliplr`) and vertical (`np.flipud`) reflections are applied to the training set to encourage rotation and reflection invariance.

---

## 4. Model Architectures & Parameter Configuration

All models are compiled using Categorical Cross-Entropy loss and optimized via Adam with a learning rate of $\alpha = 10^{-4}$ for 50 epochs, tracking validation loss for model checkpointing:

$$\mathcal{L}_{CCE} = -\sum_{i=1}^{H \times W} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

| Model | Encoder Backbone | Total Parameters | Key Architectural Features |
| :--- | :--- | :---: | :--- |
| **Vanilla U-Net** | Custom 5-stage ConvNet | ~7.8M | Symmetric encoder-decoder with skip connections, batch normalization, and dropout ($0.1$). |
| **U-Net (ResNet-34)** | ResNet-34 (Pre-trained) | **21.8M** | Residual blocks with identity shortcuts mitigating vanishing gradients; strong edge feature extraction. |
| **U-Net (VGG-16)** | VGG-16 (Pre-trained) | **138.0M** | Stacked $3\times3$ convolutions; very high parameter capacity. |
| **U-Net (Inception-V3)**| Inception-V3 (Pre-trained) | **23.8M** | Multi-branch asymmetric convolutions capturing features across diverse spatial scales. |

---

## 5. Experimental Results & Performance Benchmarking

Performance was evaluated using pixel-wise classification metrics on unseen test imagery:

* **Overall Accuracy (OA):** Proportion of correctly classified pixels across all classes.
* **Cohen's Kappa ($\kappa$):** Inter-rater agreement taking into account chance agreement.
* **Jaccard Index (mIoU):** Macro-averaged intersection over union representing spatial overlap quality.

### 5.1 Comprehensive Benchmark Evaluation

| Rank | Model Architecture | Overall Accuracy | Cohen's Kappa ($\kappa$) | Jaccard Score (mIoU) |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 | **U-Net + ResNet-34** | **76.32%** | **0.6284** | **52.80%** |
| 🥈 | **Vanilla U-Net** | 73.01% | 0.5711 | 50.53% |
| 🥉 | **U-Net + VGG-16** | 71.04% | 0.5316 | 45.81% |
| 4 | **U-Net + Inception-V3** | 69.45% | 0.5118 | 38.73% |

### 5.2 Single-Scene Detailed Evaluation

| Model | Overall Accuracy | Cohen's Kappa ($\kappa$) | Jaccard Score (mIoU) |
| :--- | :---: | :---: | :---: |
| **U-Net + ResNet-34** | **73.39%** | **0.5799** | **0.5004** |
| **Vanilla U-Net** | 70.18% | 0.5222 | 0.4757 |
| **U-Net + VGG-16** | 68.52% | 0.4900 | 0.4374 |
| **U-Net + Inception-V3** | 66.60% | 0.4690 | 0.3670 |

---

## 6. Analytical Discussion & Findings

1. **Residual Connections Enhance Spatial Boundaries:**  
   **ResNet-34** achieved the highest accuracy ($76.32\%$) and mIoU ($52.80\%$). The residual identity mappings facilitate gradient flow across deep layers, allowing the network to preserve fine spatial details essential for delineating linear road segments and building edges.

2. **Parameter Over-Capacity (VGG-16 Bottleneck):**  
   Despite having **138M parameters** (over $6\times$ more than ResNet-34), VGG-16 produced inferior results ($45.81\%$ mIoU). For remote sensing datasets with finite training patches, excessive parameters without residual skip paths can induce optimization difficulties and sub-optimal generalization.

3. **Multi-Scale Feature Blurring (Inception-V3):**  
   Inception-V3 demonstrated the lowest segmentation fidelity ($38.73\%$ mIoU). While its multi-kernel inception modules perform well in standard object classification, the concatenation of disparate receptive fields appears to introduce spatial blurring at boundaries, degrading dense pixel-level localization.

4. **Efficiency of Lightweight Custom Architectures:**  
   The **Vanilla U-Net** model (~7.8M parameters) ranked second overall, surpassing both VGG-16 and Inception-V3. This highlights the effectiveness of direct encoder-to-decoder skip connections when trained end-to-end on domain-specific aerial imagery.
