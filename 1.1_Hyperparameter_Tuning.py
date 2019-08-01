#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:23:29 2019

@author: chase.kusterer
"""


# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Importing data
sydney_df = pd.read_excel('sydney.xlsx')



###############################################################################
# Preparation
###############################################################################

sydney_data = sydney_df.loc[: , ['cartime', 'carcost', 'traincost']]
sydney_target =  sydney_df.loc[: , 'choice']


X_train, X_test, y_train, y_test = train_test_split(
            sydney_data,
            sydney_target,
            test_size = 0.25,
            random_state = 508,
            stratify = sydney_target)



###############################################################################
# Hyperparameter Tuning with Logistic Regression
###############################################################################


logreg = LogisticRegression(C = 1.0,
                            solver = 'lbfgs')


logreg_fit = logreg.fit(X_train, y_train)


logreg_pred = logreg_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))


"""
Prof. Chase:
    The hyperparameter C helps is regularize our model. In general, this means
    that if we increase C, our model will perform better on the training data
    (good if a model is underfit). Also, if we decrease C, our model will
    perform better on the testing data (good if a model is overfit).
"""


########################
# Adjusting the hyperparameter C to 100
########################

logreg_100 = LogisticRegression(C = 100,
                                solver = 'lbfgs')


logreg_100_fit = logreg_100.fit(X_train, y_train)


logreg_pred = logreg_100_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_100_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_100_fit.score(X_test, y_test).round(4))



########################
# Adjusting the hyperparameter C to 0.000001
########################

logreg_000001 = LogisticRegression(C = 0.000001,
                                solver = 'lbfgs')


logreg_000001_fit = logreg_000001.fit(X_train, y_train)


# Let's compare the testing score to the training score.
print('Training Score', logreg_000001_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_000001_fit.score(X_test, y_test).round(4))




########################
# Plotting each model's coefficient magnitudes
########################


fig, ax = plt.subplots(figsize=(8, 6))

plt.plot(logreg.coef_.T,
         'o',
         label = "C = 1",
         markersize = 12)

plt.plot(logreg_100.coef_.T,
         '^',
         label = "C = 100",
         markersize = 12)

plt.plot(logreg_000001.coef_.T,
         'v',
         label = "C = 0.001",
         markersize = 12)



plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.hlines(0, 0, X_train.shape[1])
plt.ylim(-.11, .11)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()

plt.show()

