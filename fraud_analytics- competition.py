# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:42:59 2018

@author: JThatha
"""

#importing the libraries

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble  import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split

#importing the dataset

data=pd.read_excel("D:\\Final\\ML\\Fraud Analytics\\Dataset.xlsx")
data2=data
#checking correlation between features
cor=data.corr()


#data preprocessing

    
#Employee span transformation
data2['Employment span'].replace('n/a', '0', inplace=True)    
data2['Employment span'].replace(['9 years', '< 1 year', '2 years', '10+ years', '5 years', '8 years','7 years', '4 years', 'n/a', '1 year', '3 years', '6 years'],[9,1,2,10,5,8,7,4,0,1,3,6],inplace=True)
data2['Employment span']=pd.to_numeric(data2['Employment span'])    

#Last Pay transformation
data2['Last pay'].replace('th week', '',regex=True, inplace=True)    
data2['Last pay'].replace('NA', '0',regex=True, inplace=True)   
data2['Last pay']=pd.to_numeric(data2['Last pay'])

#tenure transformation
data2['Tenure'].replace('months', '',regex=True, inplace=True)   
data2['Tenure']=pd.to_numeric(data2['Tenure'])

cor=data2.corr()


#identifying unwanted features
cols=['Customer id','Batch ID','Address 2','Address 1','Sub grade']

dummy_cols=['Grade','Home ownership','Verification of income','Payment plan']

dummy_cols2=['loan application status','Loan Application type','Joint customer verification status']

lagre_unique_values=['occupation','Reason']

most_null=['Months since last derogatory','Duration since last crime','Duration since last record']



#dropping unwanted columns
data2=data2.drop(cols,axis=1)

#dropping large unique values
data2=data2.drop(lagre_unique_values,axis=1)

#dropping most null value features
data2=data2.drop(most_null,axis=1)

#creating dummies for less distinct vlaues
data2=pd.get_dummies(data=data2,columns=dummy_cols)
data2=pd.get_dummies(data=data2,columns=dummy_cols2)


data2.isnull().sum()

null_values=data2[data2['Total amount'].isnull()]

#dropping null rows
data2=data2.drop(data2[data2['Crime details'].isnull()].index)


#make null values in R util as 0 for now
data2.loc[data2[data2['R util'].isnull()].index,'R util']=0

#dropping Collections 12months columns since collection column already exist
data2=data2.drop('Collections 12months',axis=1)

#replacing null value with mean value in Total amount,Total balance,Total credit limit
data2.loc[data2[data2['Total amount'].isnull()].index,'Total amount']=np.mean(data2['Total amount'])
data2.loc[data2[data2['Total balance'].isnull()].index,'Total balance']=np.mean(data2['Total balance'])
data2.loc[data2[data2['Total credit limit'].isnull()].index,'Total credit limit']=np.mean(data2['Total credit limit'])

#checking correlation after data preprocessing
cor=data2.corr()

#summary of dataframe
#desc=data2_norm.describe()

#data3 without target variable
data3=data2.loc[:,data2.columns!='Status of Loan']
#Normalize data in data3
data3_norm = (data3 - data3.mean()) / (data3.max() - data3.min())

cor_norm=data3_norm.corr()

#splitting train and test data
x=data3_norm
y=data2.loc[:,data2.columns=='Status of Loan']
X_train, X_test, Y_train, Y_test =train_test_split(x,y, test_size = 0.33, random_state = 5)


#fitting logistic model
model=LogisticRegression()
model.fit(X_train,Y_train)

#predicting the Y_test
Y_pred = model.predict(X_test)

#finding accuracy using confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)

print("Accuracy:",((cnf_matrix[0,0]+cnf_matrix[1,1])/Y_pred.shape[0])*100)

data3_norm.isnull().any()


model=RandomForestClassifier()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)


#Applying dimension reduction method PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_


import xgboost as xgb

model=xgb.XGBClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
