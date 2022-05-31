# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:54:36 2019

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('pos-neg.csv')

#oshin is shuffling the order of rows
dataset=dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,88].values

#onehotencoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
number=LabelEncoder()
y=number.fit_transform(y)
enc=OneHotEncoder()
y=y.reshape(-1,1)
y=enc.fit_transform(y).toarray()

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)





#splitting dataset into training and test test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#fitting simple multiple logistic regression FS required
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)


#fitting MLkNN FS required

Y=[[1],[2],[3],[4],[5],[6],[7],[8],[9]]
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer(classes=(1,2,3,4,5,6,7,8,9))
mlb_res=mlb.fit_transform(Y)
classifier = MLkNN(ignore_first_neighbours=0,k=10,s=1.0)

#fitting kNN FS required
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=7)


#fitting svm FS needed
from sklearn.svm import SVC
classifier=SVC(kernel='linear', random_state=0)

#fitting naive bayes FS required
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()

#fitting decision tree no need to FS
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)

#fitting random forests FS is required
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0)

#fitting and predicting test set results
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_pred,y_test)

#aplying cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

#Backward elimination with p value and r squared
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
	   numVars = len(x[0])
	   temp = np.zeros((367,88)).astype(int)
	   for i in range(0, numVars):
	       regressor_OLS = sm.OLS(y, x).fit()
	       maxVar = max(regressor_OLS.pvalues).astype(float)
	       adjR_before = regressor_OLS.rsquared_adj.astype(float)
	       if maxVar > SL:
	           for j in range(0, numVars - i):
	               if (regressor_OLS.pvalues[j].astype(float) == maxVar):
	                   temp[:,j] = x[:, j]
	                   x = np.delete(x, j, 1)
	                   tmp_regressor = sm.OLS(y, x).fit()
	                   adjR_after = tmp_regressor.rsquared_adj.astype(float)
	                   if (adjR_before >= adjR_after):
	                       x_rollback = np.hstack((x, temp[:,[0,j]]))
	                       x_rollback = np.delete(x_rollback, j, 1)
	                       print (regressor_OLS.summary())
	                       return x_rollback
	                   else:
	                       continue
	   regressor_OLS.summary()
	   return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87]]
X_Modeled = backwardElimination(X_opt, SL)

#fitting after backward elimintion
#splitting dataset into training and test test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_Modeled,y,test_size=0.25, random_state=0)
#fitting svm FS needed
from sklearn.svm import SVC
classifier=SVC(kernel='poly',degree=10, random_state=0)
#fitting and predicting test set results
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_pred,y_test)
