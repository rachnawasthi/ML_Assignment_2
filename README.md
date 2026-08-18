Breast Cancer Classification using Multiple ML Models

**a. Problem Statement**

The objective of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is malignant (0) or benign (1) using the Breast Cancer Wisconsin dataset.

The project demonstrates an end-to-end machine learning workflow including:

Data preprocessing

Model training

Performance evaluation

Model comparison

Deployment using Streamlit

**b. Dataset Description**

Dataset Name: Breast Cancer Wisconsin Dataset

Source: UCI Machine Learning Repository (available via sklearn)

Total Instances: 569

Number of Features: 30 numerical features

Target Variable:

0 → Malignant

1 → Benign

The dataset contains computed features from digitized images of fine needle aspirates of breast masses. All features are numeric and represent characteristics such as radius, texture, perimeter, area, smoothness, and symmetry.

**c. Models Used and Evaluation Metrics**

The following six classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble Boosting Model)

Each model was evaluated using:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

**Model Comparison Table**


| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.98     | 0.9962 | 0.9722    | 1.00   | 0.9859 | 0.9526 |
| Decision Tree            | 0.94     | 0.9381 | 0.9706    | 0.9429 | 0.9565 | 0.8608 |
| KNN                      | 0.94     | 0.9943 | 0.9444    | 0.9714 | 0.9577 | 0.8554 |
| Naive Bayes              | 0.98     | 0.9981 | 0.9722    | 1.00   | 0.9859 | 0.9526 |
| Random Forest (Ensemble) | 0.96     | 0.9943 | 0.9714    | 0.9714 | 0.9714 | 0.9048 |
| XGBoost (Ensemble)       | 0.96     | 0.9848 | 0.9714    | 0.9714 | 0.9714 | 0.9048 |


**Observations on Model Performance**

| ML Model Name            | Observation about model performance                                                                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved the highest accuracy (0.98) with perfect recall (1.0), meaning it correctly identified all benign cases in the test set. High AUC (0.9962) indicates excellent class separability. This model performs very well because the dataset is nearly linearly separable. |
| Decision Tree            | Achieved lower accuracy (0.94) and MCC (0.8608) compared to other models. Being a single tree model, it is more prone to overfitting and sensitive to small data variations. However, it provides good interpretability.                                                    |
| KNN                      | Produced high AUC (0.9943) but slightly lower accuracy (0.94). Performance depends heavily on proper feature scaling and distance calculation. It performs well but may be sensitive to noise in data.                                                                      |
| Naive Bayes              | Delivered performance almost identical to Logistic Regression with high accuracy (0.98) and perfect recall (1.0). Despite assuming feature independence, it works effectively on this dataset.                                                                              |
| Random Forest (Ensemble) | Improved stability over Decision Tree by combining multiple trees. Achieved balanced precision and recall with good MCC (0.9048). Reduced overfitting compared to a single tree model.                                                                                      |
| XGBoost (Ensemble)       | Achieved strong ensemble performance with balanced precision and recall (0.9714). Although slightly lower accuracy than Logistic Regression and Naive Bayes, it provides robust predictive capability due to boosting mechanism.                                            |


**Deployment**

The project was deployed using Streamlit Community Cloud, providing an interactive web interface that allows:

Test dataset upload (CSV)

Model selection

Real-time performance metric display

Confusion matrix visualization
