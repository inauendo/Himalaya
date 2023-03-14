# Himalaya Project

## Introduction
The Himalaya Project is a personal project in which I attempted to model and predict the success chances of mountain climbing expeditions in the Himalaya region.

The goal of the project was to use data science methods to identify the most influential factors for the success of an expedition. Despite the technological advances of the last century, extreme mountain climbing remains a dangerous endeavour, and this project is an attempt to use data-driven techniques to find optimal expedition conditions. A 'successful' expedition is defined as an expedition which has reached its goal, i.e. summited the chosen peak via at least one of the planned routes. If several routes were planned for the same expedition, the expedition is considered a success if the climbing teams succeeded on any route. If accidents took place during descent, the expedition is still considered a success.

## Data
All data for this project was taken from [The Himalayan Database](https://www.himalayandatabase.com/), a "compilation of records for all expeditions that have climbed in the Nepal Himalaya". The data is available to the public free of charge via their software. I extracted the complete data and converted it to .csv files which can be found in the /data directory. The Database stores entries for expeditions from 1905 to 2022, with extesive information on expedition conditions, members and literary references.

## Methods
As a first step, I conducted some exploratory data analysis, which can be found in the file "EDA.py". The code plots several histrograms, scatter and bar plots to examine trends in the dataset. Initially, a modelling approach based on a random forest classifier was chosen. However, due to the large amount of categorical variables in the dataset, the decision trees performed quite badly. The model was replaced by a support vector machine, which increased performance (measured in ROC AUC). Due to the difficulty in interpreting SVMs, the model was again changed to a logistic classifier. The logistic model performed almost as good as the SVM, but offered simpler interpretation by examining the weighting coefficients. Thus, the SVM approach was abandoned completely.

The Logistic classifier takes the following expedition data into account:
 - year
 - season
 - which peak was climbed
 - whether the peak was traversed
 - wheter skiing and/or paragliding was employed
 - the amount of expedition members
 - the amount of hired personnel
 - whether oxygen was used during the ascent, descent or for sleep
 
Some data cleaning had to be done to remove entries which were unrecognized or where essential data (member count) was missing.
The fields for the peak and season were encoded using one-hot encoding. To make the model more complex, polynomial features (only up to degree 2, as the model started overfitting) were created for the member and hired personnel counts.
Stratified K-fold cross-validation was used to find the optimal regularization parameter for the classifier. The regularization was set to a L2 penalty, in an attempt to increase the model's predictive power. Generally, the model achieved ROC AUC scores around 0.72.

## Results
### EDA

### Classification

## Conclusion
