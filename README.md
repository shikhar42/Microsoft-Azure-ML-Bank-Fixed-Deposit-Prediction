# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used in this project contains information about Bank's Marketing Data. The dataset has 21 columns and about 33000 rows. The goal is to predict if the client will subscribe to a fixed term deposit(y). This is one of the sample dataset provided by azure and the dataset can be found at https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

The best performing model was VotingEnsemble with the accuracy of 0.9175 using AutoML. Hence, in this case we could see that by using autoMl, we could get a more accurate model than the logistic regression model we were using in hyperdrive.

## Scikit-learn Pipeline
The Scikit-learn pipeline used in this project is as follows:
 
  * First, the data was loaded from TabularDatasetFactory
  * In the data cleaning step, rows having null values were dropped and one hot encoding was performed for categorical variables
  * Data was split into training and test datasets with a ratio of 70% and 30%, respectively
  * After that Logistic Regression Model was used for training with hyperparameter tuning such as C and max_iter using HyperDrive. I have chosen Random Sampling to choose the   hyperparameters
  * BanditPolicy was used for early termination when the objective is reached so that the hyperparameter run can stop and resources can be saved
  * Next, we found the best model based on the hyperparameters 
  * Finally, the best model was saved

I used Random Sampling to choose the hyperparameters as it is good for getting some values of hyperparameters that one cannot guess intuitively. Also, random sampling would mean that we will cover most of the sample space and will get the best model.

The bandit policy is used for early termination so that the process can be stopped once the objective is reached and resources are not wasted. It takes into the account the slack factor and evaluation_interval. This means that it terminates any run that doesn't reach the slack factor of the evaluation metric with respect to the best performing run.


## AutoML
The purpose of automl is to provide us with the best possible model by running all model and hyperparameter combinations by running it against the dataset. In our case, voting ensemble gave us the best accuracy of 91.78%
The initial preprocessing steps are similar to the ones that we used for hyperdrive run. We import the dataset using TabularDataFactory and then clean the dataset using the function created in train.py. Next, we create an automl config where we input the timeout time, algorithm type(classification), number of cross validations, dataset, primary metric and the target column. We have chosen accuracy as our primary metric  and the number of cross validations were 5.

## Pipeline comparison
Firstly, in hyperdrive only one model was used(logistic regression) on different hyperparameters. In Automl, we generated multiple models and all of them were optimized on different hyperparamters as well. The primary metric of both the methods were accuracy. In the Logistic regression model developed with hyperdrive, we got an accuracy of 90.95% whereas with automl, the best model we got was Voting Ensemble with an accuracy of 91.78%. Talking about the execution time, the hyperdrive took about 5 mins to run whereas the automl took about half an hour which was expected since the automl ran for 32 models. 

For different types of dataset, same models perform differently. Automl helps in deciding which is the best model for a specific dataset. On the other hand, automl takes time and utilizes more resources. Hence, if we know the model that we want to use, its always best to save some time and space(resources) and use hyperdrive to optimize the model we want to work on using different hyperparameters.
## Future work

  * While working on the Sklearn pipeline, it was evident from the ouput that logistic regression is a great mdodel for this dataset as the difference between the accuracies from logistic regression and the automl was very less. If a linear model like logistic is able to learn the data so well, I believe that this model can be optimized by doing some more pre processing like feature selection, impalance data handling, etc and also trying more hyperparameters. 
  * Here we used accuracy as a primary metric to compare the two techniques. In future, we can try comparning based on different primary metrics(for ex. AUC) as well as accuracy alone isn't a good estimator to assess the model's performance as we have an imbalanced dataset.

## Proof of cluster clean up

https://github.com/shikhar42/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/deletecomp.PNG
