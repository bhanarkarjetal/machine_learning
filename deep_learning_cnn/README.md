# Deep learning final project:

## Aim:
The goal is to build a vision pipeline that can read still images of hand pose signs for the digits 0 – 9 and classify them correctly.
This involves:
1.	Loading & structuring the raw data (NumPy arrays supplied by Kaggle). 
2.	Explored the distribution and quality of the images.
3.	Pre-processing (normalization, augmentation, reshaping).
4.	Evaluation & error analysis to understand where and why misclassifications occur.

## Dataset:
The dataset is taken from Kaggle which consists of images for digits in sign language. It consists of 2062 images of size 64x64 and a label dataset of digits 0 to 9 in .npy format
* x.shape = (2062, 64, 64)
* y.shape = (2062, 10)

## Data Cleaning and Exploration

1. Generated a count plot for label distribution in the entire dataset

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/digit_counts.png)

3. Plotted the first and last sample to sanity check label alignment.

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/sample_images.png)

3. Generated one sample image of each label.

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/label_images.png)

4. Visualized intra-class variation: 10 random images for a single digit.

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/sample_single_label_images.png)

## Training model on three variations of deep learning

**Variation 1 – Baseline CNN**
•	Sequential model with four hidden layers
•	Stack of three × Conv2D (3 × 3 kernel, ReLU, same padding)
•	Each convolution followed by MaxPooling2D(pool_size=(2, 2))
•	Feature maps flattened, then one Dense layer (128 units, ReLU)
•	Dropout(0.2) applied to the dense layer for regularisation
•	Output Dense(10, activation='softmax') for digit probabilities

**Variation 2 – Baseline + Data Augmentation**
•	Architecture identical to Variation 1
•	Input pipeline uses on-the-fly augmentation:
  - rotation_range = 20°
  - width_shift_range = height_shift_range = 0.20
  - shear_range = 0.20, zoom_range = 0.20
  - horizontal_flip=True, vertical_flip=True
  - fill_mode='nearest'

**Variation 3 – All Convolutional Network**
•	Removes all MaxPooling2D and Dropout layers
•	Feature maps flattened, then one Dense layer (128 units, ReLU) followed by Dense(10, softmax) for classification

## Key Findings – Model Performance Comparison
**Model 1 – CNN with MaxPooling + Dropout**
  - Test loss: 0.34
  - Test accuracy: 91.28 %
  - Train loss: 0.34
  - Train accuracy: 91.28 %
  - Best performing configuration; balanced bias–variance.

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/accuracy_1.png)

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/loss_1.png)

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/predictions_1.png)

**Model 2 – Same CNN with Data Augmentation Pipeline**
  - Test loss: 2.15
  - Test accuracy: 26.15 %
  - Train loss: 2.15
  - Train accuracy: 24.72 %
  - Heavy augmentation without re-tuning hyperparameters led to underfitting and a sharp accuracy drop.

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/accuracy_augmentation_2.png)

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/loss_augmentation_2.png)

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/predictions_2.png)

**Model 3 – All Conv (Stride 2) Network, No Pooling/Dropout**
  - Test loss: 1.31
  - Test accuracy: 77.24 %
  - Train loss: 0.24
  - Train accuracy: 95.45 %
  - High train accuracy vs. lower test accuracy indicates over-fitting despite larger capacity.

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/accuracy_3.png)

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/loss_3.png)

![alt text](https://github.com/bhanarkarjetal/deep_learning_cnn/blob/main/predictions_3.png)

## Model recommendation
* Given the observed metrics, Model 1 (the baseline CNN), which combines MaxPooling and Dropout, should be adopted as the production model. It delivers the highest test set accuracy (91.3 %) with a low, matching train and test loss, indicating a healthy balance between bias and variance. 
*	In contrast, Model 2 underfits badly after aggressive data augmentation, while Model 3 overfits, achieving very high training accuracy on unseen data. 
*	Model 1’s architecture is also computationally lighter than the all-convolutional variant and requires no additional tuning to stabilise heavy augmentations.
*	For these reasons, it offers the most reliable performance for the complexity trade-off. It provides a solid baseline that can be incrementally improved through targeted hyperparameters or data-centric tweaks without risking drastic accuracy swings.

## Future Steps
* Hyperparameter tuning:
    - Experiment with optimisers (AdamW, RMSprop, Nadam) and learning rate schedulers (Cosine decay, One Cycle).
    - Adjust kernel counts, filter sizes, and dropout rates to balance capacity and regularisation.
*	Network depth & architecture search
    - Add additional convolutional blocks or residual connections to capture finer spatial cues.
*	Transfer learning & compute reduction
    - Load a pretrained ImageNet backbone, freeze lower layers, and fine-tune only the classifier head; this cuts training time while boosting accuracy.
*	Regularisation techniques
    - Introduce batch normalisation and weight decay.

