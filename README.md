# Breast Cancer Prediction 🔬💻

This repository contains a machine learning model for predicting the presence of breast cancer using the sklearn datasets load_breast_cancer. The model is trained using Random Forest, SVM, and XGBoost algorithms. The data preprocessing, exploratory data analysis (EDA), feature selection, and oversampling using SMOTE are also included. 📊📈

## Data Features 📊

The data consists of 569 samples with 30 features each. The features are real-valued and represent characteristics of the breast cancer cells. The target variable is binary, indicating the Malignant (1 : presence) or Benign (0 : absence) of breast cancer.

The features are:

1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)
11. radius error (mean absolute error in radius lengths)
12. texture error (mean absolute error in gray-scale values)
13. perimeter error
14. area error
15. smoothness error
16. compactness error
17. concavity error
18. concave points error
19. symmetry error
20. fractal dimension error
21. worst radius
22. worst texture
23. worst perimeter
24. worst area
25. worst smoothness
26. worst compactness
27. worst concavity
28. worst concave points
29. worst symmetry
30. worst fractal dimension

## Data Preprocessing 🧪

Before training the model, the data was preprocessed using the following steps:

1. Data cleaning: Missing values were checked and handled appropriately.
2. Feature selection: Feature selection was performed to select the most relevant features for the model. The features were ranked based on their importance score from the Random Forest algorithm. The top 10 features were selected for the model.
3. Feature scaling: The features were scaled using standard scaling to ensure comparability.
4. Data splitting: The data was split into a training set and a testing set using a 70:30 ratio.

## Exploratory Data Analysis (EDA) 📈

EDA was performed using the following techniques:

1. Descriptive statistics: Basic statistics like mean, median, mode, and standard deviation were calculated for each feature.
2. Correlation matrix: A correlation matrix was created to visualize the relationships between the features.
3. Visualization: Histograms and box plots were used to visualize the distribution of the features.

## Model Training 📊

The model was trained using the following algorithms:

1. Random Forest: The Random Forest algorithm was used to train the model. The model was trained using a 70:30 train-test split.
2. SVM: The Support Vector Machine algorithm was used to train the model. The model was trained using a 70:30 train-test split.
3. XGBoost: The eXtreme Gradient Boosting algorithm was used to train the model. The model was trained using a 70:30 train-test split.

## Oversampling ⚙️

To handle the class imbalance in the dataset, the SMOTE (Synthetic Minority Over-sampling Technique) algorithm was used. The SMOTE algorithm generated synthetic samples for the minority class (breast cancer) to balance the class distribution.

## User App 📱

A user-friendly app was created to use the model for prediction. The app takes the values of the 10 selected features as input and predicts the probability of breast cancer for the given input.

To use the app, follow these steps:

1. Download the repository.
2. Set the command prompt path to the downloaded folder.
3. Run `pip install -r requirements.txt` to install the necessary packages.
4. Run `python app.py` to start the app.

The app will prompt you to enter the values of the 10 selected features. After submitting the values, the app will display the result.
