# Credit Risk Classification with Supervised Machine Learning

# Overview

For this project I was asked to train two machine learning models to classify loans as 'healthy' or 'high-risk'. One model uses a basic logistic regression, while the second oversamples the training data, then runs the logistic regression. Finally, I was asked to analyze and report on both models, and to suggest which would suit a lending company's needs better. That report is available in this repository as report.md. 

# Instructions

This project requires the following inputs, which are already included in the Jupyter Notebook code:
  
  import numpy as np
  
  import pandas as pd
  
  from pathlib import Path
  
  from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
  
  from sklearn.linear_model import LogisticRegression
  
  from imblearn.over_sampling import RandomOverSampler

Be sure that these libraries are already installed on your local machine.

# Features

This code includes a relatively straightforward use of standard machine learning tools, such as LogisticRegression and RandomOverSampler. The data is read in to the Jupyter Notebook, split between features and labels, then further split into training and testing data. The training data is then fitted to the LogisticRegression, then the regression is asked to predict y, using X_test data. The sklearn.metrics library is finally used to print the outputs of the code, including the model's balanced accuracy, confusion matrix, and classification report.

The second model uses many of the same tools, but first, it uses the RandomOverSampler from imblearn to resample the X and y training data.

# Sources

Data for this dataset was generated by edX Boot Camps LLC, and is intended for educational purposes only.

# Author

Daniel Adamson
