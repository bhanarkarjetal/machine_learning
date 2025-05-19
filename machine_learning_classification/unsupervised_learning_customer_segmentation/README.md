# Customer Segmentation- Unsupervised Learning

This project applies unsupervised machine learning techniques to segment customers based on behavioral and demographic features. The aim is to group customers into distinct clusters and analyze which segments are more likely to churn (leave) or stay.

## Dataset:
- Contains over 50,000 rows and 8 features.
- Includes a mix of demographic and behavioral customer attributes.

## Key Steps:
1. **Data Preparation**
   
   - Importing essential libraries
   - Handling missing values and cleaning the dataset
   - Exploratory Data Analysis (EDA) for pattern and outlier detection
   - Visualizations to understand feature distribution
     
3. **Clustering Algorithm Applied**
   
   - K-Means Clustering
   - Agglomerative Clustering
   - K-Means with PCA (Principal Component Analysis)
   - DBSCAN (Density-Based Spatial Clustering of Application with Noise)
     
5. **Visualization of Clusters**

   - Total Price Spending Patterns:
       - Total spending per weekday to identify shopping trends across the week
       - Spending distribution across months and years to detect seasonality or temporal trends
       - Spending behavior across daytime segments (e.g. morning, afternoon and evening)
    - Country-wise Spending Analysis:
        - Total price distribution across different countries to highlight geographical differences in customer spending
    - Sales Volume Analysis
        - Number of units sold across different daytime sements, weekdays, months and years to observe high and low sales periods
    - Units Sold by Country:
        - Comparison of units sold across countries to understand regional demand and customer engagement
    
7. **Evaluation Metrics**
   
   - Silhouette Score to evaluate the quality of Clusters
  
## Result Summary:

|Model                    |  Silhouette Score|
|-------------------------|------------------|
|K-Means                  |  0.20            |
|-------------------------|------------------|
|Agglomerative Clustering |  0.32            |
|-------------------------|------------------|
|K-Means with PCA         |  0.40            |
|-------------------------|------------------|
|DBSCAN                   |  0.22            |


## Observations:

- K-Means with PCA achieved the highest Silhouette Score (0.40), indicating relatively better clustering algorithm
- Agglomerative Clustering performed second-best witha a silhouette score of 0.32
- DBSCAN and plain K-Means had lower silhouette scores, suggesting weaker cluster compactness
- Overall, clustering performance was moderate, implying that the data has some separability but may benefit from additional feature engineering or dimensionality reduction

## Label Interpretation:

- After cluster labeling, domain analysis and comparison of features helped interpret which group likely represents churning vs non-churning customers.
- Additional analysis such as cross-referencing cluster features with churn indicators can enhance label interpretation.

## Suggestions for Improvement:

- Experimenting with different values of n_clusters to find optimal cluster separation.
- Applying PCA transformation to Agglomerative Clustering and DBSCAN for potential improvement in performance.
- Considering feature selection techniques or adding engineered features to better capture customer behavior patterns.

## Conclusion:

The project demonstrates how various clustering techniques can be applied to segment customers effectively. While K-Means with PCA yielded the best result in this scenario, further tuning and domain knowledge can significantly improve segmentation quality.
