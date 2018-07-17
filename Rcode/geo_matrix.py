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

result=pd.read_csv('/Users/hyeongcheolpark/Desktop/DSSG/gitscripper/DSSG-2018_Housing/Rcode/2018-06-splitted.csv')
result=pd.read_csv('2018-06-splitted.csv',encoding='latin-1')

geo.matrix=pd.DataFrame(squareform(pdist(df.iloc[:, 1:])), columns=df.CITY.unique(), index=df.CITY.unique())



#Example.
In [12]: df
Out[12]:
  CITY   LATITUDE   LONGITUDE
0    A  40.745392  -73.978364
1    B  42.562786 -114.460503
2    C  37.227928  -77.401924
3    D  41.245708  -75.881241
4    E  41.308273  -72.927887

In [13]: from scipy.spatial.distance import squareform, pdist

In [14]: pd.DataFrame(squareform(pdist(df.iloc[:, 1:])), columns=df.CITY.unique(), index=df.CITY.unique())
Out[14]:
           A          B          C          D          E
A   0.000000  40.522913   4.908494   1.967551   1.191779
B  40.522913   0.000000  37.440606  38.601738  41.551558
C   4.908494  37.440606   0.000000   4.295932   6.055264
D   1.967551  38.601738   4.295932   0.000000   2.954017
E   1.191779  41.551558   6.055264   2.954017   0.000000