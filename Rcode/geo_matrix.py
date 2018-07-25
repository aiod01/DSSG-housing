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

#import the dataframe
file='/Users/hyeongcheolpark/Desktop/DSSG/gitscripper/DSSG-2018_Housing/results/raw/deDuplicated_OnURL_20180717.csv'
result=pd.read_csv(file,encoding='latin-1')

#temporary deleting NA values.
result=result.dropna(subset=['lat', 'long'])

#From the location, make a square symmetric data frame(geo matrix) and name it geo_frame.
pairdist=pdist(result.loc[:,['lat','long']])
dist_matrix=squareform(pairdist)
geo_frame=pd.DataFrame(dist_matrix, columns=result.url.unique(), index=result.url.unique())

#set the lower triangle elements as 100.
geo_frame=geo_frame.where(np.triu(np.ones(geo_frame.shape)).astype(np.bool))
geo_frame=geo_frame.fillna(100)

#we will pick only pairs of urls which is bigger than 0
x=list(geo_frame[geo_frame < 0.1].stack().index)



#results = {}
#for k,v in x:
#    results.setdefault(k, []).append(v)
    
#under certain thresholds, I would like to pick the list of lists. 

#temp_set=set()
#threshold=0.3
#for i in range(geo_frame.shape[0]-1):
#    for j in range(i+1,geo_frame.shape[1]):
#        if geo_frame.iloc[i,j] <= threshold:
#            temp_subset=set()
#            temp_subset.add(result.url[i])
#            temp_subset.add(result.url[j])#What's wrong with result.url[71]?
#        temp_set=temp_subset|temp_set
    
#temp_set.add(temp_set)

#A new way, less computational method 
        
#def checkthld(i,threshold):
    #i is each row
    #j is each column. 
#    if i<j: #for each row, we look at elements whose indices bigger than the index of row.
        #give me the indics of columns for each row
#    return(#dict of key: i and value: many j values)
	#i wanna return dict of key: the row, and value: the columns 

#geo_frame.apply(checkthld, axis=1,args=(threshold,)) 
#b is a Series, deleting redundant lower triangle values.
#b=geo_frame.mask(np.triu(np.ones(geo_frame.shape)).astype(bool)).stack()

#from collections import OrderedDict, defaultdict
#b.to_dict(OrderedDict)
#x is a list of tuples, but including redundant values 


#This is to make a dict(key and value, but redundant values included)
