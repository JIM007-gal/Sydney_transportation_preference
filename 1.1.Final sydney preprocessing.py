#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:13:09 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Hult University/Academics/Dual Degree/Module B2/Machine Learning/Class 4/2nd_sydney

Purpose: To analyze the sydney  Dataset in order to publish an 
         intelligence report on Github.

Index: 
    * Data preprocessing: flagging and imputing nas, visual EDA, 
    flagging outliers, and transform categorical variables into dummies.
"""

# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil


# Loading dataset
sydney_df = pd.read_excel('0.sydney.xlsx')

sydney_df.head()
sydney_df.shape
sydney_df.info()
sydney_df_describe = sydney_df.describe()



###############################################################################
## 1. Data Preprocessing
###############################################################################

#######################################
# 1.1. Converting 'choice' to binary
#######################################

# Renaming
binary = {'CAR': 0,
          'TRAIN': 1}

sydney_df['choice'].replace(binary, inplace = True)
 


#######################################
# 1.2. Visual EDA
#######################################

# Creating folder in current directory to store graphs
cwd = os.getcwd()

if os.path.exists(cwd + '/1.2.EDA Graphs'):
  shutil.rmtree(cwd + '/1.2.EDA Graphs')

os.makedirs(cwd+'/1.2.EDA Graphs')
graph_path = cwd+'/1.2.EDA Graphs/'


# 1.1.1. Histograms numerical variables
f, axes = plt.subplots(2, 3, figsize = (10, 8))
for i, e in enumerate(list(sydney_df.columns)):
    sns.distplot(sydney_df[e],
                 bins = 'fd',
                 kde = True,
                 rug = False,
                 ax = axes[i // 3][i % 3])
    plt.xlabel(e)
    
plt.savefig(graph_path + '1.Histograms.png')


## Saving df in excel
sydney_df.to_excel('1.3.Sydney preprocessed.xlsx')