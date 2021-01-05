# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about customers of a certain bank as it relates to a marketing campaign. We seek to predict if a customer will subscribe to a product.

The best model was a StackEnsemble Model with an accuracy of 0.9168 that was obtained using AutoML vs Hyperdrive's logistic regression of max iterations 82 and C of 1 that had an accuracy of 0.91162.

To improve the AutoML score, we could increase the amount of time so that many more algorithms and approaches can be tried.
To improve HyperDrive, increase the range of hyperparameters that are to be tested.


## Scikit-learn Pipeline
The data was first cleaned and encoded then split into a training and validation set.

Scikit Learn's Logistic Regression algorithm was the classifier that was chosen.
The hyperparameters that were to be tuned were: 
 `_C_      : Inverse of regularization strength. Smaller values cause stronger regularization
 _max_iter_: Maximum number of iterations to converge`


We used RandomSampling because it would likely converge faster than other sampling methods.
For early stooping, we used BanditPolicy as it would take less time by stopping runs that are out of the defined slack.

## AutoML
The best performing model was a StackEnsembleClassifier with one of its metaleaners as a logistic regression model.The metalearner used cross-validation version.

## Pipeline comparison
The best model was a StackEnsemble Model with an accuracy of 0.9168 that was obtained using AutoML vs Hyperdrive's logistic regression of max iterations 82 and C of 1 that had an accuracy of 0.91162. Since ensembling using many models to make a decision, as compared to hyperdrives one logistic regression, it was able to achieve a higher score.

## Future work
Advanced Feature engineering. deleting and creating new features and using other methods such as target encoding could help improve the results. The resulting featureset will be able to map well to the target variable resulting into better models.

