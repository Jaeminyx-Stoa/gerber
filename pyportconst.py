# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:07:02 2021

@author: Lenovo
"""
import numpy as np

def initial_weight(returns, weight_type):
    returns = np.array(returns)
    if weight_type == 'equal':
        w = np.ones(returns.shape[1])
        w /= np.sum(w)
        
    elif weight_type == 'default':
        
        w = returns.mean(0) + abs(returns.mean(0).min())
        w /= np.sum(w)
    
    elif weight_type == 'longshort1':
        w = returns.mean(0)/np.sum(returns.mean(0))
        
    elif weight_type == 'longshort0':
        w = returns.mean(0) - returns.mean(0).mean()
        w[w>0] /= w[w>0].sum()
        w[w<0] /= w[w<0].sum()
        
    return w