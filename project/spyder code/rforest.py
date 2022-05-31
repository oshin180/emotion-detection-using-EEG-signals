# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 23:15:23 2018

@author: Admin
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('emotiondb.csv')