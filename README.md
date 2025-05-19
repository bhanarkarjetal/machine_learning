# **Machine Learning Projects Repository**

Welcome to my Machine Learning repository! This collection contains diverse projects demonstrating a range of machine learning techniques—from traditional supervised learning to deep learning with convolutional neural networks (CNNs). Each project includes real-world datasets, detailed evaluation, and model comparisons.

## **Project Summaries**

1. **Deep Learning:** CNN for Hand Pose Digit Recognition
   A deep learning pipeline using Convolutional Neural Networks to classify static images of hand gestures representing digits (0–9).
   **Data:** NumPy image arrays from Kaggle
     
   ### **Key steps:**
   - Data exploration & preprocessing (normalization, augmentation, reshaping).
   - Evaluation and error analysis.
   - Comparison of 3 CNN variations:
       1. **Model 1**: Baseline CNN with MaxPooling and Dropout – Best model (91.28% accuracy).
       2. **Model 2:** CNN with Data Augmentation – Underfit (26.15% accuracy).
       3. **Model 3:** All Convolutional Network (no pooling/dropout) – Overfit (77.24% accuracy).

   ### **Recommendation:**
     Model 1 offers the best performance–efficiency balance.

   ### **Future Work:**
     - Hyperparameter tuning
     - Network architecture enhancement
     - Transfer learning
     - Advanced regularization

2. **Machine Learning Classification Projects**
   Projects focused on both supervised and unsupervised classification tasks using real-world datasets.

   ### **Projects:**
   | Project Name | Description | Algorithms Used | Key Results |
   | Loan Approval Classification	| Predict loan status using demographic & financial features | Logistic Regression, Decision Tree, Random Forest	| Best: Decision Tree |
   | Customer Segmentation Classification	| Segment users and identify churn | K-Means, Agglomerative Clustering, PCA, DBSCAN	| Best: K-Means with PCA |

   ### **Key features:**
   - Real-world binary and multi-class classification
   - Data preprocessing (missing values, encoding, scaling)
   - Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Silhouette Score
   - Handling imbalanced datasets

   ### **Tools and Libraries:**
   - `Python`, `Jupyter`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

3. **Machine Learning Regression Projects:**
   Projects applying regression techniques to predict continuous target variables
   
   ### **Projects:**
   | Project Name | Description | Techniques Used | Evaluation |
   |--------------|-------------|-----------------|------------|
   | Mobile price prediction	| Predict mobile prices using hardware specs (RAM, CPU, etc.)	 | Log/Sqrt/Box-Cox Transformations, Polynomial Features	| MSE, R²|
   | Stock Price Prediction	| Forecast stock closing prices using historical price & volume data | Scaling, Polynomial Features (deg 2), Lasso, Ridge Regression		| MSE, R² |

   ### **Tools and libraries:**
   - `Python`, `Jupyter`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

   

