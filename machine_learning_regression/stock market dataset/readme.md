# Stock Market Closing Price Prediction & Model Comparison

This project involves predicting the closing price of a stock using different transformations and regularization techniques. The dataset consists of stock market data with features like the opening price, low, high, adjusted closing price, and volume. Various techniques such as linear regression, polynomial features, and regularization (Lasso and Ridge) are applied, and their performance is compared.

## Dataset

The dataset consits of 1200 rows which contains the following features:
- **Date**: Date of the stock market data (day, month, year)
- **Opening Price**: The price at which the stock opened
- **Low**: The lowest price of the stock on that day
- **High**: The highest price of the stock on that day
- **Adjusted Closing Price**: Adjusted closing price, accounting for splits/dividends
- **Volume**: The number of shares traded

**Target Variable**:
- **Closing Price**: The closing price of the stock

## Libraries Used

The following libraries were used to perform data manipulation, visualization, modeling, and evaluation:
1. `pandas`
2. `numpy`
3. `matplotlib`
4. `seaborn`
5. `sklearn`
6. `scipy`

## Steps Involved

1. **Data Gathering**:
   - The dataset was collected from a reliable source (Kaggle) and loaded into the project for analysis.

2. **Data Preprocessing**:
   - Data was cleaned and formatted for analysis.

3. **Data Transformation**:
   - The date was split into day, month, and year to analyze temporal patterns.
   - Necessary transformations on features like opening price, low, high, and volume were applied.

4. **Linear Regression Model**:
   - A simple linear regression model with standard scaling was trained to predict the closing price using the available features.

5. **Scaling and Polynomial Features of Degree 2**:
   - Features were scaled using standard scaling techniques.
   - Polynomial features of degree 2 were generated to capture non-linear relationships.

6. **Lasso and Ridge Regularization with Cross-Validation**:
   - Regularization techniques (Lasso and Ridge) were applied with cross-validation to find the optimal value of the regularization parameter alpha.

7. **Model Evaluation**:
   - Errors (such as MSE and R² score) were calculated for each model.
   - The performance of each model was evaluated using training and testing datasets.

8. **Error Comparison**:
   - MSE values for the models were compared between the training and testing datasets to ensure that the model generalizes well.

9. **Coefficient Comparison**:
   - Coefficients from the linear regression, Lasso, and Ridge models were compared to identify the most impactful features.

## Results

- The R² score of the final model is **99%**, indicating that the model explains 99% of the variance in the closing price.
- The Mean Squared Error (MSE) values for the training and testing datasets were very close, showing that the model is not overfitting and performs well on unseen data.

## Future Work

- Exploring more advanced models such as decision trees, random forests, and gradient boosting.
- Expanding the dataset to include more features such as external economic indicators.
- Experimenting with different regularization techniques and hyperparameter tuning.
