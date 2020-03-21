# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:40:13 2020

@author: vw178e
"""

import pandas as pd
#import matplotlib.pyplot as plt
company_data = pd.read_csv("C:/Training/Analytics/Decison_Tree/Company_data/Company_data.csv")
company_data_ori = company_data
company_data.head()

decriptive=company_data.describe()

import numpy as np
import seaborn as sns

# =============================================================================
# Data Manipulation
# =============================================================================

#dummy variables
company_dummies = pd.get_dummies(company_data[["ShelveLoc","Urban","US"]])

company_data = pd.concat([company_data,company_dummies],axis=1)
company_data = company_data.drop(['ShelveLoc','Urban','US','ShelveLoc_Bad','Urban_No','US_No'],axis=1)


#Creating a seperate dataframe which has only continuous variables
company_data_conti = company_data[['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']].copy()


#Creating a seperate dataframe which has only categorical variables
#company_data_cat = company_data[['ShelveLoc_Good','ShelveLoc_Medium','Urban_Yes','US_Yes']].copy()
company_data_cat = company_data_ori[['ShelveLoc','Urban','US']].copy()


# =============================================================================
# Exploratory Data Analysis
# =============================================================================

###Exploratory Data Analysis for categorical variables

descriptive_cat = company_data_cat.describe()

#Looking at the different values of distinct categories in our variable.

company_data_cat['ShelveLoc'].unique()
company_data_cat['Urban'].unique()
company_data_cat['US'].unique()


#No of unique categories 

len(company_data_cat['ShelveLoc'].unique())
len(company_data_cat['Urban'].unique())
len(company_data_cat['US'].unique())

#Counting no of unique categories without any missing values

company_data_cat['ShelveLoc'].nunique() 
company_data_cat['Urban'].nunique()
company_data_cat['US'].nunique()

# No of missing values

company_data_cat['ShelveLoc'].isnull().sum()
company_data_cat['Urban'].isnull().sum()
company_data_cat['US'].isnull().sum()

##Count plot / Bar Plot

sns.countplot(data = company_data_cat, x = 'ShelveLoc')
sns.countplot(data = company_data_cat, x = 'Urban')
sns.countplot(data = company_data_cat, x = 'US')

len(company_data_cat.columns)


#Exploratory data analysis for continous variables

###Exploratory Data Analysis for continous variables
descriptive_conti = company_data_conti.describe()

# Correlation matrix 
company_data_conti.corr()


# getting boxplot of price with respect to each category of gears 

heat1 = company_data_conti.corr()
sns.heatmap(heat1, xticklabels=company_data_conti.columns, yticklabels=company_data_conti.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(company_data_conti)




#==============================================================================
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(company_data.iloc[:,0:])

# =============================================================================
# 
# =============================================================================




company_data['Sales_Category'] = pd.cut(x=company_data['Sales'], bins=[0,5,9,20], labels=['Low', 'Medium','High'], right=False)
company_data['Sales_Category'].unique()
company_data.Sales_Category.value_counts()

colnames = list(company_data.columns)
predictors = colnames[1:12]
target = colnames[12]

# Splitting company_data into training and testing company_data set


# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#company_data['is_train'] = np.random.uniform(0, 1, len(company_data))<= 0.75
#company_data['is_train']
#train,test = company_data[company_data['is_train'] == True],company_data[company_data['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company_data,test_size = 0.2)

#from sklearn.tree import  DecisionTreeClassifier
#help(DecisionTreeClassifier)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")

help(RandomForestClassifier)

model.fit(train[predictors],train[target])
model = model.fit(train[predictors],train[target])
#Training company_data


train_preds = model.predict(train[predictors])
pd.Series(train_preds).value_counts()
pd.crosstab(train[target],train_preds)

# Accuracy = train
np.mean(train.Sales_Category == model.predict(train[predictors]))

#Testing company_data

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = Test
np.mean(preds==test.Sales_Category) 


# =============================================================================
# from sklearn import tree
# #tree.plot_tree(model1.fit(iris.company_data, iris.target)) 
# 
# clf = tree.DecisionTreeClassifier(random_state=0)
# clf = clf.fit(train[predictors], train[target])
# tree.plot_tree(clf) 
# 
# =============================================================================
#cluster_labels=preds
#iris_train = train
#iris_train['clust']=cluster_labels # creating a  new column and assigning it to new column 
#iris_train = iris_train.iloc[:,[5,0,1,2,3,4]]
#iris_train.head()
#
#cluster_labels=preds
#iris_test = test
#iris_test['Predicted Species']=cluster_labels # creating a  new column and assigning it to new column 
##iris_test = iris_test.iloc[:,[5,0,1,2,3,4]]
#iris_test.head()

