# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:31:10 2020

@author: subham
"""

import pandas as pd

dataset=pd.read_csv('heart_data.csv')


a=['cp', 'fbs', 'restecg', 'exang', 'slopw', 'ca', 'thal', 'target']
for i in a:
    dataset[1].unique()