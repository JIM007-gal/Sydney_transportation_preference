#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:05:20 2019

@author: chase.kusterer

Purpose: To practice ensemble modeling.
"""


# Loading new libraries
from sklearn.ensemble import RandomForestClassifier

# Loading other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



########################
# Data Preparation
########################
sydney_df = pd.read_excel('sydney_binary.xlsx')


sydney_data = sydney_df.iloc[: , :-1]
sydney_target =  sydney_df.iloc[: , -1: ]


X_train, X_test, y_train, y_test = train_test_split(
            sydney_data,
            sydney_target.values.ravel(),
            test_size = 0.25,
            random_state = 508,
            stratify = sydney_target)



###############################################################################
# Random Forest in scikit-learn
###############################################################################

# Following the same procedure as other scikit-learn modeling techniques

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)



# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)


"""
Prof. Chase:
    Although these results are good, it appears that the model may be
    overfit. Perhaps we can tweak some of the parameters to reslove this
    issue.
"""


########################
# Feature importance function
########################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

########################
        
plot_feature_importances(full_gini_fit,
                         train = X_train,
                         export = False)



plot_feature_importances(full_entropy_fit,
                         train = X_train,
                         export = False)



"""
Prof. Chase:
    When modeling with logistic regression, we noticed that traintime was not
    significant based on its p-value. Let's try removing it and see if it helps
    our random forest models.
"""


########################
# Repreparing Data
########################
sydney_df = pd.read_excel('sydney_binary.xlsx')


sydney_data = sydney_df.loc[: , ['cartime',
                                  'carcost',
                                  'traincost']]

sydney_target =  sydney_df.iloc[: , -1: ]


X_train, X_test, y_train, y_test = train_test_split(
            sydney_data,
            sydney_target.values.ravel(),
            test_size = 0.25,
            random_state = 508,
            stratify = sydney_target)



########################
# Running with Gini
########################

# Full forest using gini
sig_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
sig_fit = sig_forest_gini.fit(X_train, y_train)



# Scoring the gini model
print('Training Score', sig_fit.score(X_train, y_train).round(4))
print('Testing Score:', sig_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_sig_train = sig_fit.score(X_train, y_train)
gini_sig_test  = sig_fit.score(X_test, y_test)


"""
Prof. Chase:
    Removing traintime actually hurt the model. This is because random forest,
    unlike classical statistical models such as logistic regression, does not
    make the same assumptions about the significance of a variable.
"""



########################
# Parameter tuning with GridSearchCV
########################

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))



########################
# Building Random Forest Model Based on Best Parameters
########################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 16,
                                    n_estimators = 600,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)


###############################################################################
# Gradient Boosted Machines
###############################################################################

"""
Prof. Chase:
    Gradient boosted machines (gbms) are like decision trees, but instead of
    starting fresh with each iteration, they learn from mistakes made in
    previous iterations.
"""

from sklearn.ensemble import GradientBoostingClassifier

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )


"""
Prof. Chase:
    Notice above that we are using friedman_mse as the criterion. Friedman
    proposed that instead of focusing on one MSE value for the entire tree,
    the algoirthm should localize its optimal MSE for each region of the tree.
"""


gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)


"""
Prof. Chase:
    It appears the model is not generalizing well. Let's try to work on that
    using GridSearhCV.
"""


########################
# Applying GridSearchCV
########################

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}



# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)



# Creating a GridSearchCV object
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv = 3)



# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))



########################
# Building GBM Model Based on Best Parameters
########################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.1,
                                      max_depth = 5,
                                      n_estimators = 100,
                                      random_state = 508)



gbm_optimal.fit(X_train, y_train)


gbm_optimal_score = gbm_optimal.score(X_test, y_test)


gbm_optimal_pred = gbm_optimal.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))


gbm_optimal_train = gbm_optimal.score(X_train, y_train)
gmb_optimal_test  = gbm_optimal.score(X_test, y_test)



########################
# Saving Results
########################

# Saving best model scores
model_scores_df = pd.DataFrame({'RF_Score': [full_forest_cv.best_score_],
                                'GBM_Score': [gbm_grid_cv.best_score_]})


model_scores_df.to_excel("Ensemble_Model_Results.xlsx")



# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': rf_optimal_pred})


model_predictions_df.to_excel("Random_Forest_Model_Predictions.xlsx")


