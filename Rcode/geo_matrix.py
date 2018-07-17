#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 18:50:05 2018

@author: hyeongcheolpark
"""
import numpy as np
import scipy as sp
import pandas as pd
from scipy.spatial.distance import squareform, pdist

#How to import my June matrix?

result=pd.read_csv('/Users/hyeongcheolpark/Desktop/DSSG/gitscripper/DSSG-2018_Housing/Rcode/2018-06-splitted.csv',encoding='latin-1')
#result=pd.read_csv('2018-06-splitted.csv',encoding='latin-1')

geo_matrix=pd.DataFrame(squareform(pdist(result.iloc[:,[7,9]])), columns=result.ID.unique(), index=result.ID.unique())


