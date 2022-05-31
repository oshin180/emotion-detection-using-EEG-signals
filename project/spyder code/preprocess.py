# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 21:29:01 2019

@author: Admin
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('simple linear regression.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 88].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,:])
X[:,0:88] = imputer.transform(X[:,:])

#oshin is just trying to normalize
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

#oshin is shuffling the order of rows
df=dataset.sample(frac=1).reset_index(drop=True)


