# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:25:31 2018

@author: Admin
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm 
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
 

from sklearn.tree import DecisionTreeClassifier

# Load the emotion dataset
df = pd.read_csv('final copy2.csv')

np.where(np.isnan(X))
np.where(df.values >= np.finfo(np.float64).max)



# Create a list of feature names
feat_labels = ['avg_raw',	'negat_raw',	'peak_raw',	'inverse_raw',	'peakinv_raw',	'mean_raw',	'std_raw',	'skew_raw',	'kurt_raw',	'featureup_raw',	'featuredown_raw',	'avg_att',	'negat_att',	'peak_att',	'inverse_att',	'peakinv_att',	'mean_att',	'std_att',	'skew_att',	'kurt_att',	'featureup_att',	'featuredown_att',	'avg_med',	'negat_med',	'peak_med',	'inverse_med',	'peakinv_med',	'mean_med',	'std_med',	'skew_med',	'kurt_med',	'featureup_med',	'featuredown_med',	'avg_alpha',	'negat_alpha',	'peak_alpha',	'inverse_alpha',	'peakinv_alpha',	'mean_alpha',	'std_alpha',	'skew_alpha',	'kurt_alpha',	'featureup_alpha',	'featuredown_alpha',	'avg_beta',	'negat_beta',	'peak_beta',	'inverse_beta',	'peakinv_beta',	'mean_beta',	'std_beta',	'skew_beta',	'kurt_beta',	'featureup_beta',  'featuredown_beta',	'avg_delta',	'negat_delta',	'peak_delta',	'inverse_delta',	'peakinv_delta',	'mean_delta',	'std_delta',	'skew_delta',	'kurt_delta',	'featureup_delta',	'featuredown_delta',	'avg_gamma',	'negat_gamma',	'peak_gamma',	'inverse_gamma',	'peakinv_gamma',	'mean_gamma',	'std_gamma',	'skew_gamma',	'kurt_gamma',	'featureup_gamma',	'featuredown_gamma',	'avg_theta','	negat_theta',	'peak_theta',	'inverse_theta',	'peakinv_theta',	'mean_theta',	'std_theta',	'skew_theta',	'kurt_theta',	'featureup_theta',	'featuredown_theta']

# Create X from the features
X = df.iloc[:, :-1].values

# Create Y from output
Y = df.iloc[:, 88].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# Train the classifier
clf.fit(X_train, Y_train)


Y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
# Print the name and gini importance of each feature
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)
    
    # Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.15
sfm = SelectFromModel(clf, threshold=0.5)

# Train the selector

# Print the names of the most important features

for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels)
    
    
    
    clf = svm.SVC(kernel='poly', degree=8)
    #Train the model using the training sets
clf.fit(X_train, Y_train)
#Predict the response for test dataset
Y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))