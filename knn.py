# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:37:57 2023

@author: lenovo
"""

1]PROBLEM   --"glass.csv"

    
BUSINESS OBJECTIVE:-Prepare a model for glass classification using KNN...Target Variable is --'Type'



#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



#Loading the Datset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/KNN/glass.csv')

#EDA
df.isnull().sum()#For NA values
df.head()
df.tail()
df.shape
df.describe()#Mathematical Calculations



inpu=df.iloc[:,0:9]#Predictors
target=df.iloc[:,[9]]#Target

#Split the Dataset 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.3)

#Importing KNN model
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=13)
model.fit(x_train,y_train)#Fit the model

#Evaluate on Test Data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix=confusion_matrix(y_test,testpred)

#Evaluate on Train Data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
confusion_matrix=confusion_matrix(y_train,trainpred)
test_report=classification_report(y_train,trainpred)



2]PROBLEM   --"Zoo.csv"

BUSINESS OBJECTIVE:--Implement a KNN model to classify the animals in to categorie



#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/KNN/Zoo.csv')

#EDA
df.isnull().sum()#For NA values
df.head()
df.tail()
df.shape
df.describe()#Mathematical Calculations


inpu=df.iloc[:,1:17]#Predictors
inpu.head()#Target
target=df.type


#Split the Dataset 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inpu,target,test_size=0.3)

#Importing KNN model
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)#I did Hypertunning again and again,and found n_neighbors=5.. is optimum no of neighbors.. 
model.fit(x_train,y_train)#Fit the model

#Evaluate on Test Data
testpred=model.predict(x_test)
accuracy_score(y_test,testpred)
test_report=classification_report(y_test,testpred)
confusion_matrix=confusion_matrix(y_test,testpred)

#Evaluate on Train Data
trainpred=model.predict(x_train)
accuracy_score(y_train,trainpred)
confusion_matrix=confusion_matrix(y_train,trainpred)
test_report=classification_report(y_train,trainpred)