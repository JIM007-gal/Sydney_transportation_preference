#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:26:02 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Hult University/Academics/Dual Degree/Module B2/Machine Learning/Class 4/2nd_sydney

Purpose: To analyze the Sydney Dataset in order to publish an 
         intelligence report on Github.
         
Index: * Random Forest
       * Gradient Boosted Machines (GBM)
"""

# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import os
import shutil


# Load preprocessed dataset
sydney_df = pd.read_excel('1.3.Sydney preprocessed.xlsx')


# Redefine cwd
cwd = os.getcwd()



###############################################################################
## 1. Random Forest
###############################################################################

if os.path.exists(cwd + '/3.2.Random Forest'):
  shutil.rmtree(cwd + '/3.2.Random Forest')
  
os.makedirs(cwd + '/3.2.Random Forest')
RF_path = cwd + '/3.2.Random Forest/'


# Train/Test split
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)
sydney_df_target = sydney_df['choice']

X_train, X_test, y_train, y_test = train_test_split(sydney_df_data, 
                                                    sydney_df_target,
                                                    test_size = 0.25,
                                                    random_state = 992,
                                                    stratify = sydney_df_target)


# In case we don't have the test data, we apply GridSearchCV to select clf k
estimators_space = pd.np.arange(100, 1000, 100)
leaf_space = pd.np.arange(1, 100, 20)

param_grid = {'n_estimators': estimators_space,
              'min_samples_leaf': leaf_space}

RF_grid = RandomForestClassifier()
RF_grid_cv = GridSearchCV(RF_grid, param_grid, cv = 3)
RF_grid_cv.fit(X_train, y_train)

print('Best parameters:', RF_grid_cv.best_params_)
print('Cross Validation score:', RF_grid_cv.best_score_.round(4))


# Maximum score at n_estimators = 1, min_samples_leaf = 3
RF_clf = RandomForestClassifier(n_estimators = 18, 
                                min_samples_leaf = 2)
RF_clf.fit(X_train, y_train)

RF_clf_pred_probabilities = RF_clf.predict_proba(X_train)
RF_clf_pred_probabilities_test = RF_clf.predict_proba(X_test)

RF_clf_pred = RF_clf.predict(X_train)
RF_clf_pred_test = RF_clf.predict(X_test)

print('Training score:', RF_clf.score(X_train, y_train).round(4))
print('Validation score:', RF_clf.score(X_test, y_test).round(4))

accuracy_RF_clf = RF_clf.score(X_test, y_test).round(4)
accuracy_RF_clf_cv = cross_val_score(RF_clf, sydney_df_data, sydney_df_target, cv = 3).mean().round(4)
    
                    
## Creating ROC curve
# Calculate FPR and TPR for all thresholds of the classification
preds_RF = RF_clf_pred_probabilities[:,1]
fpr_RF, tpr_RF, threshold_RF = metrics.roc_curve(y_train, preds_RF)
roc_auc_RF = metrics.auc(fpr_RF, tpr_RF)

preds_RF_test = RF_clf_pred_probabilities_test[:,1]
fpr_RF_test, tpr_RF_test, threshold_RF_test = metrics.roc_curve(y_test, preds_RF_test)
roc_auc_RF_test = metrics.auc(fpr_RF_test, tpr_RF_test)

print('Training AUC:', roc_auc_RF.round(4))
print('Validation AUC:', roc_auc_RF_test.round(4))

plt.plot(fpr_RF, 
         tpr_RF, 
         'y', 
         label = 'RF AUC = %0.2f' % roc_auc_RF)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(RF_path+'1.ROC curve.png')
plt.show()  



###############################################################################
## 2. Gradient Boosting Machines
###############################################################################

if os.path.exists(cwd + '/3.3.GBM'):
  shutil.rmtree(cwd + '/3.3.GBM')
  
os.makedirs(cwd + '/3.3.GBM')
GBM_path = cwd + '/3.3.GBM/'


# Train/Test split
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)
sydney_df_target = sydney_df['choice']

X_train, X_test, y_train, y_test = train_test_split(sydney_df_data, 
                                                    sydney_df_target,
                                                    test_size = 0.25,
                                                    random_state = 992,
                                                    stratify = sydney_df_target)


# In case we don't have the test data, we apply GridSearchCV to select clf k
learning_space = pd.np.arange(0.001, 0.011, 0.001)
estimator_space = pd.np.arange(200, 401, 100)
depth_space = pd.np.arange(1,2)
leaf_space = pd.np.arange(1, 16, 15)

param_grid = {'learning_rate': learning_space,
              'n_estimators': estimator_space,
              'max_depth': depth_space,
              'min_samples_leaf': leaf_space}

GBM_grid = GradientBoostingClassifier()
GBM_grid_cv = GridSearchCV(GBM_grid, param_grid, cv = 3)
GBM_grid_cv.fit(X_train, y_train)

print('Best parameters:', GBM_grid_cv.best_params_)
print('Cross Validation score:', GBM_grid_cv.best_score_.round(4))


# Maximum score at learning_rate = 0.004, n_estimators = 400, max_depth = 1, min_samples_leaf = 1
GBM_clf = GradientBoostingClassifier(learning_rate = 0.004,
                                     n_estimators = 18, 
                                     max_depth = 1,
                                     min_samples_leaf = 1)
GBM_clf.fit(X_train, y_train)

GBM_clf_pred_probabilities = GBM_clf.predict_proba(X_train)
GBM_clf_pred_probabilities_test = GBM_clf.predict_proba(X_test)

GBM_clf_pred = GBM_clf.predict(X_train)
GBM_clf_pred_test = GBM_clf.predict(X_test)

print('Training score:', GBM_clf.score(X_train, y_train).round(4))
print('Validation score:', GBM_clf.score(X_test, y_test).round(4))

accuracy_GBM_clf = GBM_clf.score(X_test, y_test).round(4)
accuracy_GBM_clf_cv = cross_val_score(GBM_clf, sydney_df_data, sydney_df_target, cv = 3).mean().round(4)
    
                    
## Creating ROC curve
# Calculate FPR and TPR for all thresholds of the classification
preds_GBM = GBM_clf_pred_probabilities[:,1]
fpr_GBM, tpr_GBM, threshold_GBM = metrics.roc_curve(y_train, preds_GBM)
roc_auc_GBM = metrics.auc(fpr_GBM, tpr_GBM)

preds_GBM_test = GBM_clf_pred_probabilities_test[:,1]
fpr_GBM_test, tpr_GBM_test, threshold_GBM_test = metrics.roc_curve(y_test, preds_GBM_test)
roc_auc_GBM_test = metrics.auc(fpr_GBM_test, tpr_GBM_test)

print('Training AUC:', roc_auc_GBM.round(4))
print('Validation AUC:', roc_auc_GBM_test.round(4))

plt.plot(fpr_GBM, 
         tpr_GBM, 
         'y', 
         label = 'GBM AUC = %0.2f' % roc_auc_GBM)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(GBM_path+'1.ROC curve.png')
plt.show()  



###############################################################################
## 3. Model selection and predictions
###############################################################################

## Create a new file in directory
if os.path.exists(cwd + '/3.4.Results interpretability'):
  shutil.rmtree(cwd + '/3.4.Results interpretability')
  
os.makedirs(cwd + '/3.4.Results interpretability')
interp_path = cwd + '/3.4.Results interpretability/'
    

## Print and save AUC model results
print(f"""
RF test AUC: {roc_auc_RF_test.round(3)}
GBM test AUC: {roc_auc_GBM_test.round(3)}
""")

scores_auc = {'Cross validation scores': [roc_auc_RF_test.round(3),
                                          roc_auc_GBM_test.round(3)]}

auc_index = ['RF', 'GBM']

model_sel_scores = pd.DataFrame(scores_auc)
model_sel_scores.index = auc_index

model_sel_scores.to_excel(interp_path + '1.Scores AUC.xlsx')


## Create test profit curves
conversion_profit = int(input('Offer acceptance profit -> ',))
miss_loss = int(input('Declined offer cost -> ',))

# Calculate number of positives and negatives
test_pos = y_test.sum()
test_neg = len(y_test) - test_pos


# Vectors with percentage of observations targeted with campaign
testing_instances_RF = (test_pos * tpr_RF_test + test_neg * fpr_RF_test) / len(y_test)* 100
testing_instances_GBM = (test_pos * tpr_GBM_test + test_neg * fpr_GBM_test) / len(y_test)* 100


# Total profits by algorithm
test_profit_RF = test_pos * tpr_RF_test * conversion_profit - test_neg * fpr_RF_test * miss_loss
test_profit_GBM = test_pos * tpr_GBM_test * conversion_profit - test_neg * fpr_GBM_test * miss_loss


# Plots
plt.plot(testing_instances_RF, test_profit_RF,'g', label = 'RF')
plt.plot(testing_instances_GBM, test_profit_GBM,'r', label = 'GBM')

plt.title('Profit of classifiers')
plt.legend(loc = 'lower right')
plt.ylabel('Profit')
plt.xlabel('Percentage of Test instances')

plt.savefig(interp_path+'2.Profits by model.png')
plt.show() 


## NO BUDGET CONSTRAINT
# RF maximum profit
RF_index_test_nc = np.amax(np.where(test_profit_RF == max(test_profit_RF)))
RF_threshold_test_nc = threshold_RF_test[RF_index_test_nc]
RF_mp_test_nc = test_profit_RF[RF_index_test_nc]


# GBM maximum profit
GBM_index_test_nc = np.amax(np.where(test_profit_GBM == max(test_profit_GBM)))
GBM_threshold_test_nc = threshold_GBM_test[GBM_index_test_nc]
GBM_mp_test_nc = test_profit_GBM[GBM_index_test_nc]

profits_test_nc = {'RF': RF_mp_test_nc, 'GBM': GBM_mp_test_nc}
print('Optimal model:', [k for k,v in profits_test_nc.items() if v == max(profits_test_nc.values())][0])        
print('Max profits:', max(profits_test_nc.values()))


## BUDGET CONSTRAINT
budget_constraint = int(input('Buget constraint -> ',))
testing_instances = budget_constraint / miss_loss / len(y_test) * 100


# RF maximum profit
RF_index_test = np.amax(np.where(testing_instances_RF <= testing_instances))
RF_threshold_test = threshold_RF_test[RF_index_test]
RF_mp_test = test_profit_RF[RF_index_test]


# GBM maximum profit
GBM_index_test = np.amax(np.where(testing_instances_GBM <= testing_instances))
GBM_threshold_test = threshold_GBM_test[GBM_index_test]
GBM_mp_test = test_profit_GBM[GBM_index_test]

profits_test = {'RF': RF_mp_test, 'GBM': GBM_mp_test}
print('Optimal model:', [k for k,v in profits_test.items() if v == max(profits_test.values())][0])        
print('Max profits:', max(profits_test.values()))


## Predictions with new threshold
RF_pred_testf = (preds_RF_test >= RF_threshold_test).astype(int)
GBM_pred_testf = (preds_GBM_test >= GBM_threshold_test).astype(int)


## Visualizing confusion matrices
def confusion_matrix(predictions, file_name):
    labels = ['Car', 'Train']

    cm = metrics.confusion_matrix(y_true = y_test,
                                  y_pred = predictions)
    sns.heatmap(cm,
                annot = True,
                xticklabels = labels,
                yticklabels = labels,
                cmap = 'coolwarm')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix of the classifier')
    plt.savefig(interp_path+file_name)
    plt.show()

confusion_matrix(RF_pred_testf, '2.1.RF CM.png')
confusion_matrix(GBM_pred_testf, '3.1.GBM CM.png')


## Classification report
def classification_report(predictions, file_name):
    labels = ['Car', 'Train']
    report = metrics.classification_report(y_true = y_test,
                                           y_pred = predictions,
                                           target_names = labels,
                                           output_dict = True)
    
    report_df = pd.DataFrame(report)
    report_df.to_excel(interp_path+file_name)

classification_report(RF_pred_testf, '2.2.Classification Report.xlsx')
classification_report(GBM_pred_testf, '3.2.Classification Report.xlsx')



## Save profits and predictions
# Save profits with and without budget
profits_dict = {'Profits No Budget': max(profits_test_nc.values()), 
                'Profits Budget = '+str(budget_constraint): max(profits_test.values())}
profits_index = [[k for k,v in profits_test_nc.items() if v == max(profits_test_nc.values())][0],
                  [k for k,v in profits_test.items() if v == max(profits_test.values())][0]]

profits_df = pd.DataFrame(profits_dict, index = profits_index)
profits_df.to_excel(interp_path + '4.Profits results.xlsx')


# Reloading sydney_df dataset to account for all observations
sydney_df = pd.read_excel('1.3.Sydney preprocessed.xlsx')
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)

# Build dataframes to later concatenate them and compare against actual values
preds_RF_f = (RF_clf.predict_proba(sydney_df_data))[:,1]
preds_GBM_f = (GBM_clf.predict_proba(sydney_df_data))[:,1]

RF_fit_pred = pd.DataFrame({'RF': (preds_RF_f >= RF_threshold_test).astype(int)})
GBM_fit_pred = pd.DataFrame({'GBM': (preds_GBM_f >= GBM_threshold_test).astype(int)})


predictions = pd.concat([sydney_df['choice'], RF_fit_pred, GBM_fit_pred],
                        axis = 1)

predictions.to_excel(interp_path + '5.Predictions.xlsx')