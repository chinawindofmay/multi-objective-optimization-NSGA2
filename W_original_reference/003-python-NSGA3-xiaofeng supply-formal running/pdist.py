# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:16:17 2019

@author: 晓风
"""

import numpy as np
def pdist(x,y):
    x0=x.shape[0]
    y0=y.shape[0]
    xmy=np.dot(x,y.T)#x乘以y
    xm=np.array(np.sqrt(np.sum(x**2,1))).reshape(x0,1)
    ym=np.array(np.sqrt(np.sum(y**2,1))).reshape(1,y0)
    xmmym=np.dot(xm,ym)
    cos = xmy/xmmym
    return cos

a = np.array([[2,2],[3,4],[2,1],[3,4]])
b = np.array([[2,1],[6,4],[5,6],[3,5]])
result = pdist(a,b)