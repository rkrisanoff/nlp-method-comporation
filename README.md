## Accuracies

ru

| classifiers \ vectorizers | bag of words      | fast text         | word to vector    |
|---------------------------|-------------------|-------------------|-------------------|
| naive bayes               | 98.91598915989161 | 77.23577235772358 | 75.70009033423668 |
| random forest             | 95.5736224028907  | 92.32158988256549 | 94.39927732610658 |
| support vector machine    | 95.39295392953929 | 92.8635953026197  | 94.67028003613369 |

en

| classifiers \ vectorizers | bag of words      | fast text         | word to vector    |
|---------------------------|-------------------|-------------------|-------------------|
| naive bayes               | 99.20983318700614 | 80.33362598770852 | 76.82177348551362 |
| random forest             | 97.27831431079895 | 93.94205443371378 | 96.57594381035996 |
| support vector machine    | 96.22475856014047 | 94.5566286215979  | 96.7515364354697  |




<details>
  <summary>More details report </summary>

```
fitting the model by lang=ru
bag_of_words
        accuracy of MultinomialNB-> 98.91598915989161
              precision    recall  f1-score   support

           0       1.00      0.99      0.99       838
           1       0.96      0.99      0.98       269

    accuracy                           0.99      1107
   macro avg       0.98      0.99      0.99      1107
weighted avg       0.99      0.99      0.99      1107

        accuracy of RandomForestClassifier-> 95.5736224028907
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       838
           1       0.99      0.83      0.90       269

    accuracy                           0.96      1107
   macro avg       0.97      0.91      0.94      1107
weighted avg       0.96      0.96      0.95      1107

        accuracy of SVC-> 95.39295392953929
              precision    recall  f1-score   support

           0       0.95      1.00      0.97       838
           1       0.99      0.82      0.90       269

    accuracy                           0.95      1107
   macro avg       0.97      0.91      0.93      1107
weighted avg       0.96      0.95      0.95      1107

fast_text
        accuracy of MultinomialNB-> 77.23577235772358
              precision    recall  f1-score   support

           0       0.77      1.00      0.87       838
           1       0.95      0.07      0.12       269

    accuracy                           0.77      1107
   macro avg       0.86      0.53      0.50      1107
weighted avg       0.81      0.77      0.69      1107

        accuracy of RandomForestClassifier-> 92.32158988256549
              precision    recall  f1-score   support

           0       0.93      0.97      0.95       838
           1       0.90      0.77      0.83       269

    accuracy                           0.92      1107
   macro avg       0.91      0.87      0.89      1107
weighted avg       0.92      0.92      0.92      1107

        accuracy of SVC-> 92.8635953026197
              precision    recall  f1-score   support

           0       0.94      0.97      0.95       838
           1       0.89      0.81      0.85       269

    accuracy                           0.93      1107
   macro avg       0.91      0.89      0.90      1107
weighted avg       0.93      0.93      0.93      1107

word2vec
        accuracy of MultinomialNB-> 75.70009033423668
/home/drukhary/.cache/pypoetry/virtualenvs/experimental-Hok3v7Pw-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/drukhary/.cache/pypoetry/virtualenvs/experimental-Hok3v7Pw-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/drukhary/.cache/pypoetry/virtualenvs/experimental-Hok3v7Pw-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

           0       0.76      1.00      0.86       838
           1       0.00      0.00      0.00       269

    accuracy                           0.76      1107
   macro avg       0.38      0.50      0.43      1107
weighted avg       0.57      0.76      0.65      1107

        accuracy of RandomForestClassifier-> 94.39927732610658
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       838
           1       0.94      0.83      0.88       269

    accuracy                           0.94      1107
   macro avg       0.94      0.90      0.92      1107
weighted avg       0.94      0.94      0.94      1107

        accuracy of SVC-> 94.67028003613369
              precision    recall  f1-score   support

           0       0.96      0.97      0.96       838
           1       0.90      0.88      0.89       269

    accuracy                           0.95      1107
   macro avg       0.93      0.92      0.93      1107
weighted avg       0.95      0.95      0.95      1107

fitting the model by lang=en
bag_of_words
        accuracy of MultinomialNB-> 99.20983318700614
              precision    recall  f1-score   support

           0       1.00      0.99      0.99       870
           1       0.97      1.00      0.98       269

    accuracy                           0.99      1139
   macro avg       0.98      0.99      0.99      1139
weighted avg       0.99      0.99      0.99      1139

        accuracy of RandomForestClassifier-> 97.27831431079895
              precision    recall  f1-score   support

           0       0.97      1.00      0.98       870
           1       1.00      0.88      0.94       269

    accuracy                           0.97      1139
   macro avg       0.98      0.94      0.96      1139
weighted avg       0.97      0.97      0.97      1139

        accuracy of SVC-> 96.22475856014047
              precision    recall  f1-score   support

           0       0.96      0.99      0.98       870
           1       0.98      0.86      0.91       269

    accuracy                           0.96      1139
   macro avg       0.97      0.93      0.95      1139
weighted avg       0.96      0.96      0.96      1139

fast_text
        accuracy of MultinomialNB-> 80.33362598770852
              precision    recall  f1-score   support

           0       0.80      0.99      0.88       870
           1       0.86      0.20      0.33       269

    accuracy                           0.80      1139
   macro avg       0.83      0.60      0.61      1139
weighted avg       0.81      0.80      0.75      1139

        accuracy of RandomForestClassifier-> 93.94205443371378
              precision    recall  f1-score   support

           0       0.94      0.99      0.96       870
           1       0.94      0.79      0.86       269

    accuracy                           0.94      1139
   macro avg       0.94      0.89      0.91      1139
weighted avg       0.94      0.94      0.94      1139

        accuracy of SVC-> 94.5566286215979
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       870
           1       0.93      0.84      0.88       269

    accuracy                           0.95      1139
   macro avg       0.94      0.91      0.92      1139
weighted avg       0.94      0.95      0.94      1139

word2vec
        accuracy of MultinomialNB-> 76.82177348551362
              precision    recall  f1-score   support

           0       0.77      1.00      0.87       870
           1       1.00      0.02      0.04       269

    accuracy                           0.77      1139
   macro avg       0.88      0.51      0.45      1139
weighted avg       0.82      0.77      0.67      1139

        accuracy of RandomForestClassifier-> 96.57594381035996
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       870
           1       0.97      0.88      0.92       269

    accuracy                           0.97      1139
   macro avg       0.97      0.94      0.95      1139
weighted avg       0.97      0.97      0.97      1139

        accuracy of SVC-> 96.7515364354697
              precision    recall  f1-score   support

           0       0.98      0.97      0.98       870
           1       0.92      0.94      0.93       269

    accuracy                           0.97      1139
   macro avg       0.95      0.96      0.96      1139
weighted avg       0.97      0.97      0.97      1139
```
</details>


