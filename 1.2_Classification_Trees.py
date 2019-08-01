#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:23:29 2019

@author: jaime.iglesias
"""


# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
# Building Classification Trees
###############################################################################

from sklearn.tree import DecisionTreeClassifier # Classification trees

c_tree = DecisionTreeClassifier(random_state = 508)

c_tree_fit = c_tree.fit(X_train, y_train)


print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))


"""
    As with before, the trees are predicting well on the trianing data but
    are not generalizing well. We could go through a trial and error process in
    order to find an optimal depth level and/or leaf sizes, but it would be
    better to automate this part of the process. We can do so with using a
    technique called grid search.
"""



###############################################################################
# Hyperparameter Tuning with GridSearchCV
###############################################################################

from sklearn.model_selection import GridSearchCV



########################
# Optimizing for one hyperparameter
########################

# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 500)
param_grid = {'max_depth' : depth_space}


# Building the model object one more time
c_tree_1_hp = DecisionTreeClassifier(random_state = 508)



# Creating a GridSearchCV object
c_tree_1_hp_cv = GridSearchCV(c_tree_1_hp, param_grid, cv = 3)



# Fit it to the training data
c_tree_1_hp_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_1_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", c_tree_1_hp_cv.best_score_.round(4))




"""
    Let's extend our grid search to take on another
    hyperparameter: min_samples_leaf
"""



########################
# Optimizing for two hyperparameters
########################


# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 500)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space}


# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state = 508)



# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv = 3)



# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", c_tree_2_hp_cv.best_score_.round(4))



###############################################################################
# Visualizing the Tree
###############################################################################

# Importing the necessary libraries
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects



# Building a tree model object with optimal hyperparameters
c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 5,
                                        min_samples_leaf = 10)


c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)


dot_data = StringIO()


export_graphviz(decision_tree = c_tree_optimal_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train.columns)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)


# Saving the visualization in the working directory
graph.write_png("Sydney_Optimal_Classification_Tree.png")


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


plot_feature_importances(c_tree_optimal,
                         train = X_train,
                         export = True)



###############################################################################
# Write some code below to analyze cross-validation on your final tree model
###############################################################################
