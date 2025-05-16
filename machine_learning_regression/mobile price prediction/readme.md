# Mobile Phone Price Prediction using Linear Regression

This project aims to predict the price of mobile phones based on various features using a Linear Regression model. The features used in the model include battery, RAM, internal memory, thickness, CPU frequency, CPU cores, and others. The project involves data cleaning, transformation, model training, and performance evaluation.

## Features
The following features are used to predict the mobile phone prices:
1. Battery capacity
2. RAM (in GB)
3. Internal memory (in GB)
4. Thickness (in mm)
5. CPU frequency (in GHz)
6. CPU cores
7. Resolution
8. Phone pixel density
9. Rear camera
10. Front camera

## Approach

### 1. Data Cleaning
- **Outlier Detection and Removal**: Outliers were detected using statistical methods and removed to prevent distortion in model performance.
- **Column Name Uniformity**: Changed the column names to maintain uniformity for better accessibility.
- **Data Type Conversion**: Columns were converted to the appropriate data types to ensure proper model processing.

### 2. Data Preprocessing
- **Log Transformation**: Most of the features were transformed using the log function to handle skewed data distributions and improve model performance.
- **Square Root Transformation**: Features with zero values (such as RAM or CPU cores) were transformed using the square root to prevent taking the logarithm of zero.
- **Box-Cox Transformation**: Applied to the target variable (price) to stabilize variance and make the data more normal.
- **Standard Scaling**: StandardScaler was used to standardize the feature set, ensuring that each feature contributes equally to the model performance.
- **Polynomial Features**: Polynomial features were added to capture non-linear relationships between the features and the target variable.

### 3. Model
A Linear Regression model was used to predict the mobile phone price. Polynomial features and feature scaling were included to enhance the model's ability to capture patterns in the data.

### 4. Evaluation Metrics
- **Mean Squared Error (MSE)**: Used to evaluate the model's prediction accuracy and quantify the difference between actual and predicted values.
- **R-squared (R2)**: Used to assess the proportion of variance in the target variable (price) explained by the features.

### 5. Visualization
- Data visualization was used to check the overall distribution of data of each column.
- A graph was created to visualize the relationship between actual and predicted prices to better understand model performance.

## Libraries Used
This project uses the following Python libraries:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Data visualization
- `seaborn`: Statistical data visualization
- `sklearn`: Machine learning algorithms and tools
- `scipy`: Scientific and technical computing

Feel free to fork this repository and submit a pull request if you have suggestions or improvements for the project!
