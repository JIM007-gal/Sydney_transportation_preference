#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:23:29 2019

@author: chase.kusterer
"""


# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # logistic regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Importing data
sydney_df = pd.read_excel('sydney.xlsx')


###############################################################################
# Dataset Summary and Train/Test Split
###############################################################################

# General Summary of the Dataset
sydney_df.info()

sydney_df.head(n = 5)

sydney_df.describe().round(2)

sydney_df.corr().round(3)


###############################################################################
# WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK #
###############################################################################

# The following won't work because we have a categorical response variable
model = smf.logit(formula = "choice ~ cartime",
                  data = sydney_df)
 

result = model.fit()

###############################################################################
# WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK #
###############################################################################


###############################################################################
# Logistic Regression in statsmodels
###############################################################################


########################
# Converting choice to binary
########################

# Let's try to compare our data with choice
print(sydney_df['choice'])


# Creating a dictionary
binary_choice = {'CAR': 0,
                 'TRAIN': 1}


# Using the replace command with the dictionary
sydney_df['choice'].replace(binary_choice, inplace = True)

print(sydney_df)

sydney_df.to_excel('sydney_binary.xlsx')


###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################


# The following won't work because we have a categorical response variable
# model = smf.logit(formula = "choice ~ cartime",
#                 data = sydney_df)
 

# result = model.fit()


###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################



###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################


# Creating a dictionary
bad_decision =  {40 : 41,
                 41 : 'Python\'s',
                 42 : 'THE',
                 43 : 'MEANING',
                 44 : 'OF',
                 45 : 'LIFE'}


sydney_df.replace(bad_decision, inplace = True)



sydney_df = pd.read_excel('sydney.xlsx')

print(sydney_df['choice'])


# Creating a dictionary
binary_choice = {'CAR': 0,
                 'TRAIN': 1}


# Using the replace command with the dictionary
sydney_df['choice'].replace(binary_choice, inplace = True)

###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################


###############################################################################
# CHANGES to Train/Test Split
###############################################################################

sydney_data = sydney_df.iloc[ : , :-1]
sydney_target =  sydney_df.iloc[ : , -1: ]



X_train, X_test, y_train, y_test = train_test_split(
            sydney_data,
            sydney_target,
            test_size = 0.25,
            random_state = 508)



# Original Value Counts
sydney_target['choice'].value_counts()

sydney_target.sum() / sydney_target.count()



# Training set value counts
y_train['choice'].value_counts()

y_train.sum() / y_train.count()



# Testing set value counts
y_test['choice'].value_counts()

y_test.sum() / y_test.count()



# Stratification

X_train, X_test, y_train, y_test = train_test_split(
            sydney_data,
            sydney_target,
            test_size = 0.25,
            random_state = 508,
            stratify = sydney_target)



# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
sydney_train = pd.concat([X_train, y_train], axis = 1)



# Original Value Counts
sydney_target['choice'].value_counts()

sydney_target.sum() / sydney_target.count()



# Training set value counts
y_train['choice'].value_counts()

y_train.sum() / y_train.count()



# Testing set value counts
y_test['choice'].value_counts()

y_test.sum() / y_test.count()






########################
# Logistic Regression Modeling 
########################

# Biserial point correlations
sydney_df.corr().round(3)


# Modeling based on the most correlated explanatory variable
logistic_small = smf.logit(formula = """choice ~ cartime""",
                  data = sydney_train)


results_logistic = logistic_small.fit()


results_logistic.summary()



# Full model
logistic_full = smf.logit(formula = """choice ~ 
                                                carcost +
                                                traintime +
                                                traincost""",
                                                data = sydney_train)


results_logistic_full = logistic_full.fit()


results_logistic_full.summary()



# Significant model
logistic_sig= smf.logit(formula = """choice ~ cartime +
                                              carcost +
                                              traincost""",
                                              data = sydney_train)


results_logistic_sig = logistic_sig.fit()


results_logistic_sig.summary()


results_logistic_sig.pvalues

dir(results_logistic_sig)

# Other important summary statistics
print('AIC:', results_logistic_sig.aic.round(2))
print('BIC:', results_logistic_sig.bic.round(2))


"""
Prof. Chase:
    Lower values of aic and bic means the model is a better fit for the data.
"""

###############################################################################
# Developing a Classification Base with KNN
###############################################################################


"""
Prof. Chase:
    One of the many great things about working with scikit-learn is that there
    is extremely little difference between setting up regression models and
    setting up classification models.
"""


# Repreparing train/test split with the optimal model
sydney_data = sydney_df.loc[: , ['cartime', 'carcost', 'traincost']]
sydney_target =  sydney_df.loc[: , 'choice']


# This is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            sydney_data,
            sydney_target,
            test_size = 0.25,
            random_state = 508,
            stratify = sydney_target)



# Running the neighbor optimization code with a small adjustment for classification
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()



# Looking for the highest test accuracy
print(test_accuracy)



# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)



# It looks like 4 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors = 4)



# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)


###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################

"""
Prof. Chase:
    If you get a long error message when running the code above, try adding
    .values.ravel() to you code as in the code below.
"""

knn_clf_fit = knn_clf.fit(X_train, y_train.values.ravel())

###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################


# Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# Generating Predictions based on the optimal KNN model
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)



########################
## Does Logistic Regression predict better than KNN?
########################

from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression(C = 1)


logreg_fit = logreg.fit(X_train, y_train)


# Predictions
logreg_pred = logreg_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################

"""
Prof. Chase:
    The FutureWarning is trying to tell us that our model object's default
    solver is changing in its next version, meaning we might get different
    results when using it in the future.
"""

logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1)


logreg_fit = logreg.fit(X_train, y_train)


# Predictions
logreg_pred = logreg_fit.predict(X_test)

knn_clf_fit = knn_clf.fit(X_train, y_train.values.ravel())


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

###############################################################################
# ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! 
###############################################################################



########################
# Creating a confusion matrix
########################

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_pred))


# Visualizing a confusion matrix
import seaborn as sns

labels = ['Car', 'Train']

cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Inferno')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


########################
# Creating a classification report
########################

from sklearn.metrics import classification_report

print(classification_report(y_true = y_test,
                            y_pred = logreg_pred))



# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = logreg_pred,
                            target_names = labels))




###############################################################################
# Cross Validation with k-folds
###############################################################################

# Cross Validating the knn model with three folds
cv_knn_3 = cross_val_score(knn_clf,
                           sydney_data,
                           sydney_target,
                           cv = 3)


print(cv_knn_3)


print(pd.np.mean(cv_knn_3).round(3))

print('\nAverage: ',
      pd.np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))



# Cross Validating the knn model with three folds
cv_knn_5 = cross_val_score(knn_clf,
                           sydney_data,
                           sydney_target,
                           cv = 5)


print(cv_knn_5)


print(pd.np.mean(cv_knn_5).round(3))

print('\nAverage: ',
      pd.np.mean(cv_knn_5).round(3),
      '\nMinimum: ',
      min(cv_knn_5).round(3),
      '\nMaximum: ',
      max(cv_knn_5).round(3))




# Cross Validating the knn model with three folds
cv_knn_10 = cross_val_score(knn_clf,
                           sydney_data,
                           sydney_target,
                           cv = 10)


print(cv_knn_10)


print(pd.np.mean(cv_knn_10).round(3))

print('\nAverage: ',
      pd.np.mean(cv_knn_10).round(3),
      '\nMinimum: ',
      min(cv_knn_10).round(3),
      '\nMaximum: ',
      max(cv_knn_10).round(3))
