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

file='/Users/hyeongcheolpark/Desktop/DSSG/gitscripper/DSSG-2018_Housing/results/raw/deDuplicated_OnURL_20180717.csv'
result=pd.read_csv(file,encoding='latin-1')

#temporary deleting NA values.
result=result.dropna(subset=['lat', 'long'])

#result.iloc[:,[5,7]]
pairdist=pdist(result.loc[:,['lat','long']])
dist_matrix=squareform(pairdist)
geo_frame=pd.DataFrame(dist_matrix, columns=result.url.unique(), index=result.url.unique())


#under certain thresholds, I would like to pick the list of lists. 

temp_set=set()
threshold=1
for i in range(geo_frame.shape[0]-1):
    for j in range(i+1,geo_frame.shape[1]):
        if geo_frame.iloc[i,j] <= threshold:
            temp_subset={result.url[i],result.url[j]}
    temp_set=temp_subset|temp_set
temp_set.add(temp_set)

        