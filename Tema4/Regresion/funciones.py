#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

def quitarOutliers(dfNum):
    for c in dfNum.columns:
        Q1 = dfNum[c].quantile(0.25)
        Q3 = dfNum[c].quantile(0.75)
        IQR = Q3 - Q1
        maxi = Q3 + 1.5*IQR
        mini = Q1- 1.5*IQR
        dfNum.loc[dfNum[c]>maxi,c]=maxi
        dfNum.loc[dfNum[c]<mini,c]=mini
    return dfNum

