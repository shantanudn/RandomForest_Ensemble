# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:40:13 2020

@author: vw178e
"""

import pandas as pd
#import matplotlib.pyplot as plt
fraud_data = pd.read_csv("C:/Training/Analytics/Decison_Tree/Fraud_check/Fraud_check.csv")
fraud_data_ori = fraud_data
fraud_data.head()

decriptive=fraud_data.describe()

import numpy as np
import seaborn as sns

# =============================================================================
# Data Manipulation
# =============================================================================

#dummy variables
fraud_dummies = pd.get_dummies(fraud_data[["Undergrad","Marital.StatMarital.Status","Urban"]])
fraud_dummies = fraud_dummies.drop(['Undergrad_NO','Marital.StatMarital.Status_Divorced','Urban_NO'],axis=1)

fraud_data = pd.concat([fraud_data,fraud_dummies],axis=1)
fraud_data=fraud_data.drop(['Undergrad','Marital.StatMarital.Status','Urban'],axis=1)


fraud_data['Risk_Factor'] = pd.cut(x=fraud_data['Taxable.Income'], bins=[1, 30000,100000], labels=['Good', 'Risky'], right=False)
fraud_data['Risk_Factor'].unique()
fraud_data.Risk_Factor.value_counts()

colnames = list(fraud_data.columns)
predictors = colnames[1:7]
target = colnames[7]

#Creating a seperate dataframe which has only continuoMarital.Status variables
fraud_data_conti = fraud_data[['Taxable.Income','City.Population','Work.Experience']].copy()


#Creating a seperate dataframe which has only categorical variables
#fraud_data_cat = fraud_data[['Undergrad_Good','Undergrad_Medium','Urban_Yes','Marital.Status_Yes']].copy()
fraud_data_cat = fraud_data_ori[['Undergrad','Marital.StatMarital.Status','Urban']].copy()


# =============================================================================
# Exploratory Data Analysis
# =============================================================================

###Exploratory Data Analysis for categorical variables

descriptive_cat = fraud_data_cat.describe()

#Looking at the different values of distinct categories in our variable.

fraud_data_cat['Undergrad'].unique()
fraud_data_cat['Urban'].unique()
fraud_data_cat['Marital.Status'].unique()


#No of unique categories 

len(fraud_data_cat['Undergrad'].unique())
len(fraud_data_cat['Urban'].unique())
len(fraud_data_cat['Marital.Status'].unique())

#Counting no of unique categories without any missing values

fraud_data_cat['Undergrad'].nunique() 
fraud_data_cat['Urban'].nunique()
fraud_data_cat['Marital.Status'].nunique()

# No of missing values

fraud_data_cat['Undergrad'].isnull().sum()
fraud_data_cat['Urban'].isnull().sum()
fraud_data_cat['Marital.Status'].isnull().sum()

##Count plot / Bar Plot

sns.countplot(data = fraud_data_cat, x = 'Undergrad')
sns.countplot(data = fraud_data_cat, x = 'Urban')
sns.countplot(data = fraud_data_cat, x = 'Marital.Status')

len(fraud_data_cat.columns)


#Exploratory data analysis for continoMarital.Status variables

###Exploratory Data Analysis for continoMarital.Status variables
descriptive_conti = fraud_data_conti.describe()

# Correlation matrix 
fraud_data_conti.corr()


# getting boxplot of price with respect to each category of gears 

heat1 = fraud_data_conti.corr()
sns.heatmap(heat1, xticklabels=fraud_data_conti.columns, yticklabels=fraud_data_conti.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(fraud_data_conti)




#==============================================================================
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(fraud_data.iloc[:,0:])

# =============================================================================
# 
# =============================================================================




# Splitting fraud_data into training and testing fraud_data set


# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#fraud_data['is_train'] = np.random.uniform(0, 1, len(fraud_data))<= 0.75
#fraud_data['is_train']
#train,test = fraud_data[fraud_data['is_train'] == True],fraud_data[fraud_data['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(fraud_data,test_size = 0.2)

#from sklearn.tree import  DecisionTreeClassifier
#help(DecisionTreeClassifier)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")

help(RandomForestClassifier)

model.fit(train[predictors],train[target])
model = model.fit(train[predictors],train[target])
#Training fraud_data


train_preds = model.predict(train[predictors])
pd.Series(train_preds).value_counts()
pd.crosstab(train[target],train_preds)

# Accuracy = train
np.mean(train.Risk_Factor == model.predict(train[predictors]))

#Testing fraud_data

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = Test
np.mean(preds==test.Risk_Factor) 


# =============================================================================
# from sklearn import tree
# #tree.plot_tree(model1.fit(iris.fraud_data, iris.target)) 
# 
# clf = tree.DecisionTreeClassifier(random_state=0)
# clf = clf.fit(train[predictors], train[target])
# tree.plot_tree(clf) 
# 
# =============================================================================
#clMarital.Statuster_labels=preds
#iris_train = train
#iris_train['clMarital.Statust']=clMarital.Statuster_labels # creating a  new column and assigning it to new column 
#iris_train = iris_train.iloc[:,[5,0,1,2,3,4]]
#iris_train.head()
#
#clMarital.Statuster_labels=preds
#iris_test = test
#iris_test['Predicted Species']=clMarital.Statuster_labels # creating a  new column and assigning it to new column 
##iris_test = iris_test.iloc[:,[5,0,1,2,3,4]]
#iris_test.head()

