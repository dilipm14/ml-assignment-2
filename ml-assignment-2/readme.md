a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset and deploy them using Streamlit Community Cloud.
The task is to predict whether an individual’s annual income exceeds $50K based on demographic and employment-related features. This is a binary classification problem where different machine learning algorithms are evaluated using standard performance metrics.



b. Dataset Description

The dataset used in this project is the Adult Census Income Dataset obtained from the UCI Machine Learning Repository.

Dataset Details:
Total Instances: 48,842
Number of Features: 14 input features + 1 target variable
Type of Problem: Binary Classification
Target Variable: income
<=50K
>50K

Feature Examples:

age
workclass
education
marital-status
occupation
race
sex
capital-gain
capital-loss
hours-per-week
native-country
The dataset contains both numerical and categorical features. Categorical variables were encoded before training the models.



c. Models Used

The following six machine learning models were implemented on the same dataset:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (GaussianNB)
Random Forest (Ensemble – Bagging)
XGBoost (Ensemble – Boosting)


Model Performance Comparison

              Model  Accuracy       AUC  Precision    Recall        F1
Logistic Regression  0.825580  0.848023   0.705641  0.447625  0.547771   
      Decision Tree  0.808844  0.742534   0.591136  0.616135  0.603377   
                KNN  0.830646  0.849744   0.660029  0.582303  0.618735   
        Naive Bayes  0.801628  0.851748   0.662252  0.325309  0.436300   
      Random Forest  0.854752  0.903193   0.731402  0.607677  0.663824   
            XGBoost  0.870413  0.925298   0.769231  0.644112  0.701133   


Observations on Model Performance

ML Model Name	        Observation about Model Performance
Logistic Regression	Provided strong baseline performance but lower recall compared to ensemble models.
Decision Tree	        Moderate performance with balanced precision and recall but lower AUC.
KNN	                Improved performance over Decision Tree with better MCC and F1 score.
Naive Bayes	        High AUC but very low recall, indicating difficulty capturing positive class properly.
Random Forest	        Significantly improved performance with strong AUC and balanced precision-recall.
XGBoost	                Achieved the highest Accuracy (0.8719), AUC (0.9270), F1 Score (0.7179), and MCC (0.6376), making it the best     performing model.