# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:36:13 2018

@author: Admin
"""

import pandas as pd
import numpy as np


data = pd.read_csv('trydata.csv')

data=data.fillna(data.mean())

data.to_csv('PythonExport.csv', sep=',')