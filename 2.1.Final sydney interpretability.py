#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 09:12:34 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Hult University/Academics/Dual Degree/Module B2/Machine Learning/Class 4/2nd_sydney

Purpose: To analyze the Sydney Dataset in order to publish an 
         intelligence report on Github.
         
Index: * Feature Engineering and Logistic Regression
       * K Nearest Neighbors Classifier
       * Classification tree
       * Model Selection
"""

# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import os
import shutil


# Load preprocessed dataset
sydney_df = pd.read_excel('1.3.Sydney preprocessed.xlsx')


# Redefine cwd
cwd = os.getcwd()



###############################################################################
## 1. Feature Engineering and Linear Regression
###############################################################################

## Full model with statsmodels
full_model = smf.logit(formula = """choice ~ sydney_df['cartime'] +
                                             sydney_df['carcost'] +
                                             sydney_df['traincost']
                                             """,
                       data = sydney_df)

results = full_model.fit()
print(results.summary())


# Saving results in a csv file
if os.path.exists(cwd + '/2.2.Logistic Regression'):
  shutil.rmtree(cwd + '/2.2.Logistic Regression')
  
os.makedirs(cwd+'/2.2.Logistic Regression')
first_reg_path = cwd+'/2.2.Logistic Regression/'

beginning_text_1 = """documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
end_text_1 = '\end{document}'

f = open(first_reg_path + '1.Regression.csv', 'w')
f.write(beginning_text_1)
f.write(results.summary().as_csv())
f.write(end_text_1)
f.close()



#########################
# 1.1. Logistic Regression hyperparameter tunning
#########################

# Train/Test split
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)
sydney_df_target = sydney_df['choice']

X_train, X_test, y_train, y_test = train_test_split(sydney_df_data, 
                                                    sydney_df_target,
                                                    test_size = 0.25,
                                                    random_state = 992,
                                                    stratify = sydney_df_target)


# In case we don't have the test data, we apply GridSearchCV to select optimal k
c_space = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
solver_space = ['newton-cg', 'lbfgs']
param_grid = {'C': c_space,
              'solver': solver_space}

logreg_grid = LogisticRegression()
logreg_grid_cv = GridSearchCV(logreg_grid, param_grid, cv = 3)
logreg_grid_cv.fit(X_train, y_train)

print('Best parameters:', logreg_grid_cv.best_params_)
print('Cross Validation score:', logreg_grid_cv.best_score_.round(4))


# Maximum score at C = 0.001
logreg_clf = LogisticRegression(C = 0.001, solver = 'newton-cg')
logreg_clf.fit(X_train, y_train)

logreg_clf_pred_probabilities = logreg_clf.predict_proba(X_train)
logreg_clf_pred_probabilities_test = logreg_clf.predict_proba(X_test)

logreg_pred = logreg_clf.predict(X_train)
logreg_pred_test = logreg_clf.predict(X_test)

print('Training score:', logreg_clf.score(X_train, y_train).round(4))
print('Validation score:', logreg_clf.score(X_test, y_test).round(4))

accuracy_logreg_clf = logreg_clf.score(X_test, y_test).round(4)
accuracy_logreg_clf_cv = cross_val_score(logreg_clf, sydney_df_data, sydney_df_target, cv = 3).mean().round(4)


## Creating ROC curve
# Calculate FPR and TPR for all thresholds of the classification
preds_logreg = logreg_clf_pred_probabilities[:,1]
fpr_logreg, tpr_logreg, threshold_logreg = metrics.roc_curve(y_train, preds_logreg)
roc_auc_logreg = metrics.auc(fpr_logreg, tpr_logreg)

preds_logreg_test = logreg_clf_pred_probabilities_test[:,1]
fpr_logreg_test, tpr_logreg_test, threshold_logreg_test = metrics.roc_curve(y_test, preds_logreg_test)
roc_auc_logreg_test = metrics.auc(fpr_logreg_test, tpr_logreg_test)

print('Training AUC:', roc_auc_logreg.round(4))
print('Validation AUC:', roc_auc_logreg_test.round(4))

plt.plot(fpr_logreg, 
         tpr_logreg, 
         'y', 
         label = 'Logistic Regression AUC = %0.2f' % roc_auc_logreg)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(first_reg_path+'2.ROC curve.png')
plt.show()  
               


###############################################################################
## 2. k Nearest Neighbors
###############################################################################

#########################
# 2.1. Build a KNN model only with significant variables
#########################

if os.path.exists(cwd + '/2.3.KNN classifier'):
  shutil.rmtree(cwd + '/2.3.KNN classifier')
  
os.makedirs(cwd + '/2.3.KNN classifier')
knn_path = cwd + '/2.3.KNN classifier/'


# Train/Test split
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)
sydney_df_target = sydney_df['choice']

X_train, X_test, y_train, y_test = train_test_split(sydney_df_data, 
                                                    sydney_df_target,
                                                    test_size = 0.25,
                                                    random_state = 992,
                                                    stratify = sydney_df_target)

## Scaling our Train/Test split
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)
X_train = feature_scaler.transform(X_train)
X_test = feature_scaler.transform(X_test)


# In case we don't have the test data, we apply GridSearchCV to select clf k
k_space = pd.np.arange(1,21)
param_grid = {'n_neighbors': k_space}

knn_grid = KNeighborsClassifier()
knn_grid_cv = GridSearchCV(knn_grid, param_grid, cv = 3)
knn_grid_cv.fit(X_train, y_train)

print('Best parameters:', knn_grid_cv.best_params_)
print('Cross Validation score:', knn_grid_cv.best_score_.round(4))


# Maximum score at k = 18
knn_clf = KNeighborsClassifier(n_neighbors = 18)
knn_clf.fit(X_train, y_train)

knn_clf_pred_probabilities = knn_clf.predict_proba(X_train)
knn_clf_pred_probabilities_test = knn_clf.predict_proba(X_test)

knn_clf_pred = knn_clf.predict(X_train)
knn_clf_pred_test = knn_clf.predict(X_test)

print('Training score:', knn_clf.score(X_train, y_train).round(4))
print('Validation score:', knn_clf.score(X_test, y_test).round(4))

accuracy_knn_clf = knn_clf.score(X_test, y_test).round(4)
accuracy_knn_clf_cv = cross_val_score(knn_clf, sydney_df_data, sydney_df_target, cv = 3).mean().round(4)
    
                    
## Creating ROC curve
# Calculate FPR and TPR for all thresholds of the classification
preds_knn = knn_clf_pred_probabilities[:,1]
fpr_knn, tpr_knn, threshold_knn = metrics.roc_curve(y_train, preds_knn)
roc_auc_knn = metrics.auc(fpr_knn, tpr_knn)

preds_knn_test = knn_clf_pred_probabilities_test[:,1]
fpr_knn_test, tpr_knn_test, threshold_knn_test = metrics.roc_curve(y_test, preds_knn_test)
roc_auc_knn_test = metrics.auc(fpr_knn_test, tpr_knn_test)

print('Training AUC:', roc_auc_knn.round(4))
print('Validation AUC:', roc_auc_knn_test.round(4))

plt.plot(fpr_knn, 
         tpr_knn, 
         'y', 
         label = 'KNN AUC = %0.2f' % roc_auc_knn)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(knn_path+'1.ROC curve.png')
plt.show()  



###############################################################################
## 3. Decision tree with scikit-learn
###############################################################################

if os.path.exists(cwd + '/2.4.Classification tree'):
  shutil.rmtree(cwd + '/2.4.Classification tree')
  
os.makedirs(cwd + '/2.4.Classification tree')
tree_path = cwd + '/2.4.Classification tree/'


# Train/Test split
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)
sydney_df_target = sydney_df['choice']
X_train, X_test, y_train, y_test = train_test_split(sydney_df_data,
                                                    sydney_df_target,
                                                    test_size = 0.25,
                                                    random_state = 992)


# Hyperparameter tuning
criterion_space = ['gini', 'entropy']
depth_space = pd.np.arange(1,15)
leaf_space = pd.np.arange(25,50)

param_grid = {'criterion': criterion_space,
              'max_depth': depth_space,
              'min_samples_leaf': leaf_space}

tree_grid = DecisionTreeClassifier(random_state = 992)
tree_grid_cv = GridSearchCV(tree_grid, param_grid, cv = 3)
tree_grid_cv.fit(X_train, y_train)

print('Best parameters:', tree_grid_cv.best_params_)
print('Cross Validation score:', tree_grid_cv.best_score_.round(4))


# Maximum score at criterion = entropy, Max depth = 2, Min samples leaf = 38
tree_clf = DecisionTreeClassifier(criterion = 'entropy', 
                                  max_depth = 2,
                                  min_samples_leaf = 38,
                                  random_state = 992)

tree_clf.fit(X_train, y_train)

tree_clf_pred_probabilities = tree_clf.predict_proba(X_train)
tree_clf_pred_probabilities_test = tree_clf.predict_proba(X_test)

tree_clf_pred = tree_clf.predict(X_train)
tree_clf_pred_test = tree_clf.predict(X_test)

print('Training score:', tree_clf.score(X_train, y_train).round(4))
print('Validation score:', tree_clf.score(X_test, y_test).round(4))

accuracy_tree_clf = tree_clf.score(X_test, y_test).round(4)
accuracy_tree_clf_cv = cross_val_score(tree_clf, sydney_df_data, sydney_df_target, cv = 3).mean().round(4)

# Plot tree_clf
dot_data = StringIO()

export_graphviz(decision_tree = tree_clf,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = sydney_df_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png(tree_path+'1.Decision tree.png')


# Plot feature importance
def plot_feature_importance(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize = (12,9))
    n_features = train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    if export == True:
        plt.savefig(tree_path+"2.Tree feature importance.png")
        plt.show()
        
plot_feature_importance(tree_clf, train = X_train, export = True)


## Creating ROC curve
# Calculate FPR and TPR for all thresholds of the classification
preds_tree = tree_clf_pred_probabilities[:,1]
fpr_tree, tpr_tree, threshold_tree = metrics.roc_curve(y_train, preds_tree)
roc_auc_tree = metrics.auc(fpr_tree, tpr_tree)

preds_tree_test = tree_clf_pred_probabilities_test[:,1]
fpr_tree_test, tpr_tree_test, threshold_tree_test = metrics.roc_curve(y_test, preds_tree_test)
roc_auc_tree_test = metrics.auc(fpr_tree_test, tpr_tree_test)

print('Training AUC:', roc_auc_tree.round(4))
print('Validation AUC:', roc_auc_tree_test.round(4))

plt.plot(fpr_tree, 
         tpr_tree, 
         'y', 
         label = 'Decision tree AUC = %0.2f' % roc_auc_tree)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(tree_path+'3.ROC curve.png')
plt.show()



###############################################################################
## 4. Linear Discriminant Analysis
###############################################################################

if os.path.exists(cwd + '/2.5.LDA'):
  shutil.rmtree(cwd + '/2.5.LDA')

os.makedirs(cwd + '/2.5.LDA')
LDA_path = cwd + '/2.5.LDA/'


# Train/Test split
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)
sydney_df_target = sydney_df['choice']

X_train, X_test, y_train, y_test = train_test_split(sydney_df_data, 
                                                    sydney_df_target,
                                                    test_size = 0.25,
                                                    random_state = 992,
                                                    stratify = sydney_df_target)

# In case we don't have the test data, we apply GridSearchCV to select clf k
solver_space = ['svd', 'lsqr']
param_grid = {'solver': solver_space}

LDA_grid = LinearDiscriminantAnalysis()
LDA_grid_cv = GridSearchCV(LDA_grid, param_grid, cv = 3)
LDA_grid_cv.fit(X_train, y_train)

print('Best parameters:', LDA_grid_cv.best_params_)
print('Cross Validation score:', LDA_grid_cv.best_score_.round(4))


# Maximum score at solver = 'svd'
LDA_clf = LinearDiscriminantAnalysis()
LDA_clf.fit(X_train, y_train)

LDA_clf_pred_probabilities = LDA_clf.predict_proba(X_train)
LDA_clf_pred_probabilities_test = LDA_clf.predict_proba(X_test)

LDA_pred = LDA_clf.predict(X_train)
LDA_pred_test = LDA_clf.predict(X_test)

print('Training score:', LDA_clf.score(X_train, y_train).round(4))
print('Validation score:', LDA_clf.score(X_test, y_test).round(4))

accuracy_LDA_clf = LDA_clf.score(X_test, y_test).round(4)
accuracy_LDA_clf = cross_val_score(LDA_clf, sydney_df_data, sydney_df_target, cv = 3).mean().round(4)
    
                    
## Creating ROC curve
# Calculate FPR and TPR for all thresholds of the classification
preds_LDA = LDA_clf_pred_probabilities[:,1]
fpr_LDA, tpr_LDA, threshold_LDA = metrics.roc_curve(y_train, preds_LDA)
roc_auc_LDA = metrics.auc(fpr_LDA, tpr_LDA)

preds_LDA_test = LDA_clf_pred_probabilities_test[:,1]
fpr_LDA_test, tpr_LDA_test, threshold_LDA_test = metrics.roc_curve(y_test, preds_LDA_test)
roc_auc_LDA_test = metrics.auc(fpr_LDA_test, tpr_LDA_test)

print('Training AUC:', roc_auc_LDA.round(4))
print('Validation AUC:', roc_auc_LDA_test.round(4))

plt.plot(fpr_LDA, 
         tpr_LDA, 
         'y', 
         label = 'LDA AUC = %0.2f' % roc_auc_LDA)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig(LDA_path+'1.ROC curve.png')
plt.show()  



###############################################################################
## 5. Model selection and predictions
###############################################################################

## Create a new file in directory
if os.path.exists(cwd + '/2.6.Results interpretability'):
  shutil.rmtree(cwd + '/2.6.Results interpretability')
  
os.makedirs(cwd + '/2.6.Results interpretability')
interp_path = cwd + '/2.6.Results interpretability/'
    

## Print and save AUC model results
print(f"""
Logreg test AUC: {roc_auc_logreg_test.round(3)}
KNN test AUC: {roc_auc_knn_test.round(3)}
Tree test AUC: {roc_auc_tree_test.round(3)}
LDA test AUC: {roc_auc_LDA_test.round(3)}
""")

scores_auc = {'Cross validation scores': [roc_auc_knn_test.round(3),
                                          roc_auc_logreg_test.round(3),
                                          roc_auc_tree_test.round(3),
                                          roc_auc_LDA_test.round(3)]}

auc_index = ['KNN', 'logreg', 'tree', 'LDA']

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
testing_instances_logreg = (test_pos * tpr_logreg_test + test_neg * fpr_logreg_test) / len(y_test)* 100
testing_instances_knn = (test_pos * tpr_knn_test + test_neg * fpr_knn_test) / len(y_test)* 100
testing_instances_tree = (test_pos * tpr_tree_test + test_neg * fpr_tree_test) / len(y_test)* 100
testing_instances_LDA = (test_pos * tpr_LDA_test + test_neg * fpr_LDA_test) / len(y_test)* 100


# Total profits by algorithm
test_profit_logreg = test_pos * tpr_logreg_test * conversion_profit - test_neg * fpr_logreg_test * miss_loss
test_profit_knn = test_pos * tpr_knn_test * conversion_profit - test_neg * fpr_knn_test * miss_loss
test_profit_tree = test_pos * tpr_tree_test * conversion_profit - test_neg * fpr_tree_test * miss_loss
test_profit_LDA = test_pos * tpr_LDA_test * conversion_profit - test_neg * fpr_LDA_test * miss_loss


# Plots
plt.plot(testing_instances_logreg, test_profit_logreg,'y',label = 'logreg')
plt.plot(testing_instances_knn, test_profit_knn,'g', label = 'KNN')
plt.plot(testing_instances_tree, test_profit_tree,'r', label = 'tree')
plt.plot(testing_instances_LDA, test_profit_LDA,'b', label = 'LDA')

plt.title('Profit of classifiers')
plt.legend(loc = 'lower right')
plt.ylabel('Profit')
plt.xlabel('Percentage of Test instances')

plt.savefig(interp_path+'2.Profits by model.png')
plt.show() 


## NO BUDGET CONSTRAINT
# Logistic Regression profit
logreg_index_test_nc = np.amax(np.where(test_profit_logreg == max(test_profit_logreg)))
logreg_threshold_test_nc = threshold_logreg_test[logreg_index_test_nc]
logreg_mp_test_nc = test_profit_logreg[logreg_index_test_nc]

# KNN maximum profit
knn_index_test_nc = np.amax(np.where(test_profit_knn == max(test_profit_knn)))
knn_threshold_test_nc = threshold_knn_test[knn_index_test_nc]
knn_mp_test_nc = test_profit_knn[knn_index_test_nc]

# LDA maximum profit
LDA_index_test_nc = np.amax(np.where(test_profit_LDA == max(test_profit_LDA)))
LDA_threshold_test_nc = threshold_LDA_test[LDA_index_test_nc]
LDA_mp_test_nc = test_profit_LDA[LDA_index_test_nc]

profits_test_nc = {'Logistic Regression': logreg_mp_test_nc, 
                   'KNN': knn_mp_test_nc,
                   'LDA': LDA_mp_test_nc}

print('Optimal model:', [k for k,v in profits_test_nc.items() if v == max(profits_test_nc.values())][0])        
print('Max profits:', max(profits_test_nc.values()))


## Budget constraint
budget_constraint = int(input('Buget constraint -> ',))
testing_instances = budget_constraint / miss_loss / len(y_test) * 100

# Logistic Regression maximum profit
logreg_index_test = np.amax(np.where(testing_instances_logreg <= testing_instances))
logreg_threshold_test = threshold_logreg_test[logreg_index_test]
logreg_mp_test = test_profit_logreg[logreg_index_test]

# KNN maximum profit
knn_index_test = np.amax(np.where(testing_instances_knn <= testing_instances))
knn_threshold_test = threshold_knn_test[knn_index_test]
knn_mp_test = test_profit_knn[knn_index_test]

# LDA maximum profit
LDA_index_test = np.amax(np.where(testing_instances_LDA <= testing_instances))
LDA_threshold_test = threshold_LDA_test[LDA_index_test]
LDA_mp_test = test_profit_LDA[LDA_index_test]

profits_test = {'Logreg': logreg_mp_test, 'KNN': knn_mp_test, 'LDA': LDA_mp_test}
print('Optimal model:', [k for k,v in profits_test.items() if v == max(profits_test.values())][0])        
print('Max profits:', max(profits_test.values()))


## Predictions with new threshold
logreg_pred_testf = (preds_logreg_test >= logreg_threshold_test).astype(int)
knn_pred_testf = (preds_knn_test >= knn_threshold_test).astype(int)
LDA_pred_testf = (preds_LDA_test >= LDA_threshold_test).astype(int)


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

confusion_matrix(logreg_pred_testf, '3.1.Logistic Regression CM.png')
confusion_matrix(knn_pred_testf, '4.1.KNN CM.png')
confusion_matrix(tree_clf_pred_test, '5.1.Decision Tree CM.png')
confusion_matrix(LDA_pred_testf, '6.1.LDA CM.png')


## Classification report
def classification_report(predictions, file_name):
    labels = ['Car', 'Train']
    report = metrics.classification_report(y_true = y_test,
                                           y_pred = predictions,
                                           target_names = labels,
                                           output_dict = True)
    
    report_df = pd.DataFrame(report)
    report_df.to_excel(interp_path+file_name)

classification_report(logreg_pred_testf, '3.2.Classification Report.xlsx')
classification_report(knn_pred_testf, '4.2.Classification Report.xlsx')
classification_report(tree_clf_pred_test, '5.2.Classification Report.xlsx')
classification_report(LDA_pred_testf, '6.2.Classification Report.xlsx')



## Save predictions
# Save profits with and without budget
profits_dict = {'Profits No Budget': max(profits_test_nc.values()), 
                'Profits Budget = '+str(budget_constraint): max(profits_test.values())}
profits_index = [[k for k,v in profits_test_nc.items() if v == max(profits_test_nc.values())][0],
                  [k for k,v in profits_test.items() if v == max(profits_test.values())][0]]

profits_df = pd.DataFrame(profits_dict, index = profits_index)
profits_df.to_excel(interp_path + '7.Profits results.xlsx')


# Reloading sydney_df dataset to account for all observations
sydney_df = pd.read_excel('1.3.Sydney preprocessed.xlsx')
sydney_df_data = sydney_df.drop(['traintime', 'choice'], axis = 1)


# Build dataframes to later concatenate them and compare against actual values
preds_logreg_f = (logreg_clf.predict_proba(sydney_df_data))[:,1]
preds_LDA_f = (LDA_clf.predict_proba(sydney_df_data))[:,1]

logreg_fit_pred = pd.DataFrame({'logreg': (preds_logreg_f >= logreg_threshold_test).astype(int)})
tree_fit_pred = pd.DataFrame({'tree': tree_clf.predict(sydney_df_data)})
LDA_fit_pred = pd.DataFrame({'LDA': (preds_LDA_f >= LDA_threshold_test).astype(int)})

feature_scaler = StandardScaler()
feature_scaler.fit(sydney_df_data)
sydney_df_data = feature_scaler.transform(sydney_df_data)

preds_knn_f = (knn_clf.predict_proba(sydney_df_data))[:,1]
knn_fit_pred = pd.DataFrame({'KNN': (preds_knn_f >= knn_threshold_test).astype(int)})

predictions = pd.concat([sydney_df['choice'], logreg_fit_pred, knn_fit_pred, tree_fit_pred, LDA_fit_pred],
                        axis = 1)

predictions.to_excel(interp_path + '8.Predictions.xlsx')