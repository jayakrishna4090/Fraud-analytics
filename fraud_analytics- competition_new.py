# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:42:59 2018

@author: Jayakrishna
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

gb=data.groupby('Batch ID')
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

data2.to_excel("D:\\Final\\ML\\Fraud Analytics\\Modified.xlsx",index=False)
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

model=LogisticRegression()
model=RandomForestClassifier()

#finding most important features using Recurrsive feature Elimination method
from sklearn.feature_selection import RFE
rfe = RFE(model, None)
rfe=rfe.fit(X_train,Y_train)
print(rfe.support_)
print(rfe.ranking_)

#[ 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 42, 43]
x=data3_norm.iloc[:,rfe.ranking_==1]

y=data2.loc[:,data2.columns=='Status of Loan']
X_train, X_test, Y_train, Y_test =train_test_split(x,y, test_size = 0.33, random_state = 5)

#RandomForest Classifier - 83.46
model=RandomForestClassifier()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)

print("Accuracy:",((cnf_matrix[0,0]+cnf_matrix[1,1])/Y_pred.shape[0])*100)

#XGBOOST classifier --83.16
import xgboost as xgb

model=xgb.XGBClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test) 


#selecting important feature using feature importance --82.4

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train,Y_train)
print(model.feature_importances_)
Y_pred = model.predict(X_test) 


import matplotlib.pyplot as plt
import seaborn as sns

g = sns.factorplot("Grade", "Income", data=data, kind="bar", palette="muted", legend=False)

#@plt.scatter(data, data2, c=colors, cmap=cmap)
plt.scatter(x=data['Address 1'],y=data['Loan amount'])
plt.show()
"""
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
Y_pred = model.predict(X_test) """

"""
#------------------------------------------------
import pandas as pd
train=pd.read_csv("C:\\Users\\jthatha\\Desktop\\ML\\sklearn\\train.csv")
test=pd.read_csv("C:\\Users\\jthatha\\Desktop\\ML\\sklearn\\test.csv")
trainLabels=pd.read_csv("C:\\Users\\jthatha\\Desktop\\ML\\sklearn\\trainLabels.csv",names='label')
str1='col'
col=[ str1+str(each) for each in range(1,41) ]
train=pd.read_csv("C:\\Users\\jthatha\\Desktop\\ML\\sklearn\\train.csv",names=col)
test=pd.read_csv("C:\\Users\\jthatha\\Desktop\\ML\\sklearn\\test.csv",names=col)
data=pd.concat([train,trainLabels],axis=1)
data.drop([ 'a', 'b', 'e', 'l.1'],axis=1,inplace=True)
cor=data.corr()

from sklearn.cross_validation import train_test_split
X_train=data.iloc[:,0:40]
Y_train=data.iloc[:,40]
X_test=test

from sklearn.ensemble  import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
id=[]
for each in range(1,9001):
    pd.DataFrame(id.append(each))
ids=pd.DataFrame(id)
Y_pred=pd.DataFrame(Y_pred)

out=pd.concat([ids,Y_pred],axis=1)
out.to_csv(path_or_buf='C:\\Users\\jthatha\\Desktop\\ML\\sklearn\\out.csv')
"""

from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['Batch ID']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION")
plt.axis('off')
plt.show()
kiva_loans_data=data
total = kiva_loans_data.isnull().sum().sort_values(ascending = False)
percent = ((kiva_loans_data.isnull().sum()/kiva_loans_data.isnull().count())*100).sort_values(ascending = False)
missing_kiva_loans_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_kiva_loans_data


sns.distplot(kiva_loans_data['Loan amount'])
plt.show() 
plt.scatter(range(kiva_loans_data.shape[0]), np.sort(kiva_loans_data['Loan amount'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()
