# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about customers of a certain bank as it relates to a marketing campaign. We seek to predict if a customer will subscribe to a product.

The best model was a StackEnsemble Model with an accuracy of 0.9168 that was obtained using AutoML vs Hyperdrive's Logistic Regression of max iterations 82 and C of 1 that had an accuracy of 0.91162.

![Pipeline architecture](project1_architecture.png)

Logistic regresssion which is a linear method for predicting if an observation belongs to a class or not was used.

## Scikit-learn Pipeline
The data was first cleaned and encoded then split into a training and validation set.

Scikit Learn's Logistic Regression algorithm was the classifier that was chosen.
The hyperparameters that were to be tuned were: 
 ```
 _C_      : Inverse of regularization strength. Smaller values cause stronger regularization

 _max_iter_: Maximum number of iterations to converge
 ```

Since time and compute resources are limited, we cannot explore all the possible hyperparameters. We therefore need to sample the hyperparameters and set an early termination policy that will ensure we are efficient.  

* We used RandomParameterSampling because it computationally and time efficient. The differences between random and grid sampling are discussed below.

| Random Sampling | Grid Sampling
| -----------------|--------------
| Supports early termination| Supports early termination
| Can use both discrete and continous values | Can only use choice (discrete values)
| Selects randomly | Tries all possible combinations of the hyperparameters
| More computationally efficient (we have limited resources)| Less computationally efficient
| Takes less time (the lab has a time limit) | Takes more time

* According to Udacity notes, an early termination policy specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer. This means that the hyperparameter tuning will take less time overall as the run will be stopped and another hyperameter set run initiated. For early stooping, we used BanditPolicy as it would take less time by stopping runs that are out of the defined slack.

## AutoML
The best performing model was a StackEnsembleClassifier with one of its metaleaners as a logistic regression model with a max iteration value of 100 and C of 10. The metalearner used cross-validation version. AutoML also produced useful classification metrics that we didn't define and has a feature to explain the best model.

## Pipeline comparison
The best model was a StackEnsemble Model with an accuracy of 0.9168 that was obtained using AutoML vs Hyperdrive's logistic regression of max iterations 82 and C of 1 that had an accuracy of 0.91162. Since ensembling using many models to make a decision, as compared to Hyperdrive's one logistic regression, it was able to achieve a higher score.

## Future Improvement
Advanced Feature engineering. deleting and creating new features and using other methods such as target encoding could help improve the results. The resulting feature set will be able to map well to the target variable resulting into better models.

To improve the AutoML score, we could increase the amount of time so that many more algorithms and approaches can be tried.
To improve HyperDrive, increase the range of hyperparameters that are to be tested which allows for thorough tuning.

