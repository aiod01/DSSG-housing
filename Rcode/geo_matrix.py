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
import os
cwd=os.getcwd()
#How to import my June matrix?

file='/Users/hyeongcheolpark/Desktop/DSSG/gitscripper/DSSG-2018_Housing/results/deDuplicated_OnID_20180713.csv'
result=pd.read_csv(file,encoding='latin-1')


geo_matrix=pd.DataFrame(squareform(pdist(result.iloc[:,['latitude','longitude']])), columns=result.ID.unique(), index=result.ID.unique())


