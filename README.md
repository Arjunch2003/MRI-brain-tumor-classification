# MRI Brain Tumor Classification using Deep Learning

This project implements a CNN–Transformer hybrid deep learning architecture to classify brain tumors from MRI scans. The CNN component extracts rich spatial features from medical images, while the Transformer captures global contextual relationships to improve classification performance. The model automatically categorizes MRI images into multiple brain tumor types, supporting efficient and accurate medical image analysis and diagnosis assistance.

## Problem Statement

Brain tumor diagnosis using MRI scans is time-consuming and requires expert radiologists. This project aims to automate brain tumor classification using deep learning to improve efficiency and accuracy.

---

## Dataset

Source : [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Classes (4 Categories)

* Glioma
* Meningioma
* Pituitary
* No Tumor

The dataset contains labeled MRI images organized by class, suitable for supervised learning.

---

# Methodology

The proposed system follows a complete deep learning pipeline consisting of:

1. MRI image preprocessing
2. Brain region extraction
3. Dataset splitting and loading
4. Hybrid CNN–Transformer feature learning
5. Focal loss optimization
6. Model evaluation and visualization

---

# Medical Image Preprocessing Pipeline

A specialized medical image preprocessing pipeline was developed to improve MRI quality, remove noise, standardize intensity values, and focus the model on relevant anatomical structures.

## 1. Grayscale MRI Loading

All MRI scans were loaded in grayscale format using OpenCV. Since MRI images mainly contain intensity-based information, grayscale representation preserves medical features while reducing computational complexity.

---

## 2. Brain Region Extraction and Background Removal

To isolate the brain region:

* Otsu thresholding was applied to generate a binary image.
* External contours were detected.
* The largest contour was assumed to represent the brain.
* A padded bounding box was created around the brain region.
* A contour mask removed non-brain background pixels.

This step improves feature localization and reduces irrelevant information.

---

## 3. Noise Reduction

MRI scans were denoised using Non-Local Means Denoising:

* `cv2.fastNlMeansDenoising()`

This method removes noise while preserving important anatomical edges and tumor boundaries.

---

## 4. Medical Image Intensity Standardization

To handle MRI intensity variations:

* Dark background pixels were excluded.
* Mean and standard deviation were computed from brain tissue pixels only.
* Intensity outliers were clipped using the 3-sigma rule.
* Z-score normalization was applied.
* Intensities were scaled to the range [0,1].

This improves consistency across scans acquired from different sources.

---

## 5. Contrast Enhancement

CLAHE (Contrast Limited Adaptive Histogram Equalization) was applied to enhance local tissue contrast and improve tumor visibility.

Parameters used:

* Clip Limit = 2.0
* Tile Grid Size = 8 × 8

---

## 6. Morphological Refinement

Morphological closing operations were performed using a 3 × 3 kernel to:

* Fill small gaps
* Smooth segmented regions
* Improve structural continuity

---

## 7. Image Resizing

All MRI scans were resized to:

```
224 × 224
```

using Lanczos interpolation for high-quality resizing.

---

## 8. Tensor Conversion and Final Normalization

The processed MRI image was converted into a single-channel PyTorch tensor:

```
[H, W] → [1, H, W]
```

Final normalization mapped image intensities from [0,1] to [-1,1] for stable neural network training.

---

# Dataset Splitting and Data Loading

The dataset was divided into:

* Training Set
* Validation Set
* Testing Set

The training dataset was split using an 80:20 ratio.

| Dataset    | Images |
| ---------- | ------ |
| Training   | 4569   |
| Validation | 1143   |
| Testing    | 1311   |

---

## Class Distribution for Training

### Training Dataset

| Class      | Images |
| ---------- | ------ |
| Glioma     | 1069   |
| Meningioma | 1058   |
| No Tumor   | 1255   |
| Pituitary  | 1187   |

---


# Model Architecture

The proposed model combines CNN-based local feature extraction with Transformer-based global contextual learning.

The architecture consists of:

1. CNN Backbone
2. Patch Embedding Layer
3. Transformer Encoder
4. Classification Head

---

# CNN Backbone

A lightweight residual CNN backbone was designed specifically for medical image analysis.

## Initial Feature Extraction

The MRI input image:

```
1 × 224 × 224
```

is processed using:

* 7 × 7 convolution
* Batch normalization
* ReLU activation
* Max pooling

Output:

```
32 × 56 × 56
```

---

## Residual Learning Blocks

Two residual blocks were used for deeper feature extraction.

Each residual block contains:

* Two 3 × 3 convolution layers
* Batch normalization
* ReLU activation
* Dropout regularization
* Shortcut residual connection

Feature map progression:

| Layer   | Output Shape  |
| ------- | ------------- |
| Layer 1 | 64 × 28 × 28  |
| Layer 2 | 128 × 14 × 14 |

---

## Final CNN Feature Refinement

A final convolution layer increases feature depth to 256 channels followed by adaptive average pooling.

Final CNN feature map:

```
256 × 7 × 7
```

---

# Patch Embedding Layer

The CNN feature map is converted into Transformer-compatible patch embeddings.

* Patch Size = 2 × 2
* Embedding Dimension = 192

The feature map is transformed into:

```
9 patches
```

Each patch is projected into a 192-dimensional embedding vector.

Final patch embedding shape:

```
[B, 9, 192]
```

---

# Transformer Encoder

Transformer blocks capture global contextual relationships between different MRI regions.

## Positional Encoding and Class Token

* Learnable positional embeddings were added.
* A learnable class token was prepended.

Input sequence:

```
[B, 10, 192]
```

---

## Multi-Head Self-Attention

The Transformer encoder uses:

* Embedding Dimension = 192
* Attention Heads = 6
* Transformer Layers = 2

Self-attention helps the model focus on clinically important tumor regions and spatial dependencies.

---

## Feed Forward Network

Each Transformer block contains:

* Layer normalization
* Multi-head self-attention
* GELU activation
* Multi-layer perceptron (MLP)
* Residual skip connections
* Dropout regularization

---

# Classification Head

The final class token representation is passed through fully connected layers for classification.

The model predicts one of four classes:

* Glioma
* Meningioma
* Pituitary
* No Tumor

---

# Loss Function

Medical datasets often contain moderate class imbalance. To improve learning on minority and difficult samples, Focal Loss was used instead of standard cross-entropy loss.

## Focal Loss Formula

```
FL(pt) = -α(1 - pt)^γ log(pt)
```

Where:

* pt = predicted probability for the true class
* α = balancing factor
* γ = focusing parameter

Parameters used:

* α = 1.0
* γ = 2.0

---

# Learning Rate Scheduling

A custom learning rate scheduler combining:

1. Linear Warmup
2. Cosine Decay

was implemented for stable optimization.

## Warmup Phase

For the first 35% of epochs:

```
1e-6 → 2e-3
```

The learning rate gradually increases.

---

## Cosine Decay Phase

For the remaining epochs, cosine annealing gradually decreases the learning rate toward:

```
1e-6
```

This improves convergence and final model performance.

---

# Training Strategy

The model was trained using:

| Component               | Configuration |
| ----------------------- | ------------- |
| Optimizer               | AdamW         |
| Weight Decay            | 0.01          |
| Dropout                 | 0.15          |
| Gradient Clipping       | 1.0           |
| Early Stopping Patience | 7             |
| Epochs                  | 25            |

Gradient clipping improves Transformer training stability and prevents exploding gradients.

---

# Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* Weighted F1-score
* Macro F1-score
* Micro F1-score

Weighted F1-score was selected as the primary evaluation metric due to moderate class imbalance.

---

# Training Performance

## Final Training Results

| Metric              | Score  |
| ------------------- | ------ |
| Training Accuracy   | 97.94% |
| Training F1-score   | 97.94% |
| Validation Accuracy | 91.43% |
| Validation F1-score | 91.44% |

---

## Training and Validation Curves

The loss and accuracy curves demonstrate stable convergence of the hybrid CNN–Transformer architecture.

### Observations

* Training and validation loss decrease steadily.
* Validation accuracy stabilizes above 90%.
* Mild overfitting appears in later epochs but remains controlled due to dropout, focal loss, and early stopping.
* The learning rate scheduler enables smooth convergence.

---

# Testing Results

The final model was evaluated on an independent testing dataset containing 1311 MRI images.

## Overall Test Performance

| Metric            | Score  |
| ----------------- | ------ |
| Accuracy          | 91.30% |
| Weighted F1-score | 91.21% |
| Macro F1-score    | 90.82% |
| Micro F1-score    | 91.30% |
| Precision         | 91.19% |
| Recall            | 91.30% |

---

# Class-wise Performance

| Class      | Precision | Recall | F1-score |
| ---------- | --------- | ------ | -------- |
| Glioma     | 0.9155    | 0.8667 | 0.8904   |
| Meningioma | 0.8441    | 0.8137 | 0.8286   |
| No Tumor   | 0.9409    | 0.9827 | 0.9614   |
| Pituitary  | 0.9385    | 0.9667 | 0.9524   |

---

# Confusion Matrix Analysis

The confusion matrix demonstrates strong classification capability across all tumor categories.

## Key Observations

* Most predictions lie along the diagonal, indicating correct classification.
* No Tumor and Pituitary classes achieved the highest accuracy.
* Most misclassifications occurred between:

  * Glioma and Meningioma
  * Meningioma and Pituitary

This behavior is expected due to visual similarity between certain tumor structures in MRI scans.

---

# Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* tqdm

---

# Conclusion

This project demonstrates the effectiveness of combining CNN-based local feature extraction with Transformer-based global contextual learning for MRI brain tumor classification.

The proposed hybrid architecture achieved strong classification performance with:

* 91.30% testing accuracy
* 91.21% weighted F1-score
* Robust generalization on unseen MRI scans

The system shows strong potential for assisting automated medical diagnosis and computer-aided radiology applications.
