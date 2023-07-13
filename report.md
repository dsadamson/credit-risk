# Credit Risk Classification Report

## Overview of the Analysis

The purpose of this analysis is to use supervised machine learning models to evaluate lending risk. Each model was trained on 'lending_data.csv', included in the Resources folder of this repository, which included the following data for each loan: loan size, interest, borrower income, borrowers' debt-to-income ratio, borrowers' number of accounts, derogatory marks, and borrowers' total debt. Moreover, the data originally included a 'loan_status' column, which tracked whether a particular loan was marked as healthy or high-risk. Using the variable 'value_counts', I found the correct number of loans marked as healthy ('0') or high-risk ('1'). 

To train the first model, I began by splitting the data, using train_test_split from Scikit Learn. Next, I trained a LogisticRegression model from the same source. This model was fit with the split training data from the preceding cell. This LogisticRegression model was then asked to predict y, based on the features (X), which were separated after loading the .csv file. Using balanced_accuracy_scoore, confusion_matrix, and classification_report from Scikit Learn, I printed these outputs from the model, which will be explained in the 'Results' section of this report.

To train the second model, I oversampled the features data, using RandomOverSampler from IMBLearn. The RandomOverSampler model was fitted with X and y -- or features and labels -- training data. Again, the oversampled data was fitted to a LogisticRegression model, asked to make a prediction, then made to print summaries of its outputs.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

* The balanced accuracy of the first LogisticRegression model is 0.95213. This indicates, controlling for imbalances between the labels, the machine learning model could successfully predict y ninety-five percent of the time and makes errors only five percent of the time.

* The confusion_matrix function displayed the following:

          array([[14926,    75],
            [   46,   461]])

  Visually this demonstrates how many loans are true positives, true negatives, false positives, and false negatives. For example, the bottom left value in the array shows how many loans were predicted to be healthy (0) but are high-risk (1), in actuality. The array indicates that forty-six loans were incorrectly marked as healthy, when they were actually high risk, and seventy-five were marked as high-risk when they were actually healthy. Compared to the true positives and true negatives, the occurances of these false positives and false negatives are relatively few.

* The classification_report function displayed the following:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     15001
           1       0.86      0.91      0.88       507

    accuracy                           0.99     15508
   macro avg       0.93      0.95      0.94     15508
  weighted avg       0.99      0.99      0.99     15508

  This indicates that eighty-six percent of loans are correctly marked as high-risk, or fourteen percent are incorrectly marked as high-risk, with the model predicting healthy loans more accurately than high-risk loans. Nevertheless, this indicates that the model may misidentify otherwise deserving borrowers as high-risk.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

* The balanced accuracy score of the oversampled model is 0.99417. This indicates higher accuracy, controlling for imbalances between labels, for the oversampled model than the first model. Less than one percent of the labels are wrongly predicted, probably owing to the larger amount of training data available due to oversampling. 

* The confusion_matrix function displayed the following:

      array([[14915,    86],
            [    3,   504]])

  This indicates fewer false positives and false negatives than the first model. Most importantly for lenders, fewer high-risk loans are misidentified as healthy loans.

* The classification_report displayed the following:
         
                  precision    recall  f1-score   support

            0       1.00      0.99      1.00     15001
            1       0.85      0.99      0.92       507

      accuracy                           0.99     15508
    macro avg       0.93      0.99      0.96     15508
  weighted avg       1.00      0.99      0.99     15508

  The results are similar to the first model in many ways, although the oversampled model predicts high-risk loans with one percent less accuracy than the original model. This will have similar effects as the original model, predicting that otherwise deserving borrowers are high-risk but preventing high-risk borrowers from being classified as 'healthy.'
  

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

In sum, the oversampled model is most deserving of my recommendation. In many respects, it is similar to the first model, including the same drawbacks -- e.g., misidentifying 'healthy' borrowers as 'high-risk.' However, considering the oversampled model's higher balanced accuracy score and its ability to limit the number of high-risk borrowers that it misidentifies as 'healthy,' its performance would be far more beneficial to a lending services company. For some lendees, problems may persist with being misidentified as high-risk, but the lending company could trust that the risk of its loans could be limited by this model.
