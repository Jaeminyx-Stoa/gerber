# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import seaborn as sns
from scipy.optimize import minimize
import statsmodels.api as sm

from pyportcov import *
from pyportobj import *
from pyportconst import *

class WeightOptimizer:

    def __init__(self, cov=None, returns=None,
                 init_weight=None,
                 bm_weight=None, bm_bnd=None, bnd=None, gross_weight=None, net_weight=None,
                 tolcon = 1e-30, display=False):
        self.returns = returns
        self.cov = cov
        
        self.init_weight = init_weight
        self.bm_weight = bm_weight
        self.bm_bnd = bm_bnd
        self.bnd = bnd
        self.gross_weight = gross_weight
        self.net_weight = net_weight
        
        self.tolcon = tolcon
        self.display = display
        
        
    def custom_weight(self, objective, add=None):
        """
        allows generation of weights with return series (TxN) given a custom objective function
        depending on the objective function further parameters can be added
        otherwise all necessary calculations will be generated from the return series by the custom objective function
        
        can be used for: drawdown, skewness, varisk, cvarisk
        """
        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight})               
        
        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*self.returns.shape[1])
        
        w0 = self.init_weight
        if add:
            pars = [self.returns]
            pars.extend(add)
        else:
            pars = [self.returns]
        res = minimize(objective, w0, args=pars, method='SLSQP',
                       constraints=cons, bounds = bounds, options={'disp':False},
                       tol=self.tolcon)
        
        if self.display and not res.success:
            print(res.message)
        return res.x
    
    def ls1_markowitz_weight(self):
        '''
        input returns dataframe
        returns unconstrained markowitz weight - max sharpe ratio
        net exposure of 1
        cov = returns.cov() / avg_cov(returns)
        '''
        return np.dot(np.linalg.inv(self.cov), self.returns.mean())/np.sum(np.dot(np.linalg.inv(self.cov), self.returns.mean()))
        
    def markowitz_weight_bnd_sr(self):
        '''
        input: returns dataframe
        cov = returns.cov() / avg_cov(returns)
        target = targeted sharpe ratio (this is not annualized)
        (returns, cov, init_weight, bm_weight, bm_bnd, gross_weight, target_sr)
        (returns, cov, init_weight, bnd, gross_weight, target_sr)
        '''
        V = np.array(self.cov) #covariance
        R = self.returns.mean()
    
        w0 = self.init_weight

        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight})  
            
        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*self.returns.shape[1])
        
        res = minimize(sharpe_objective, w0, args=[R,V], method='SLSQP',
                  constraints=cons, bounds = bounds, options={'disp': False},
                  tol=self.tolcon)
        if self.display and not res.success:
            print(res.message)
        return res.x
    
    def markowitz_weight_bnd_mu(self, target_mu):
        '''
        input: returns dataframe
        cov = returns.cov() / avg_cov(returns)
        target = targeted sharpe ratio (this is not annualized)
        (returns, cov, init_weight, bm_weight, bm_bnd, gross_weight, target_mu)
        (returns, cov, init_weight, bnd, gross_weight, target_mu)
        '''
        V = np.array(self.cov) #covariance
        R = self.returns.mean()
    
        w0 = self.init_weight

        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight},
                {'type': 'eq', 'fun': lambda x: (x*R).sum()-target_mu})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                    {'type': 'eq', 'fun': lambda x: (x*R).sum()-target_mu})  

        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*self.returns.shape[1])
        
        res= minimize(std_objective, w0, args=[V], method='SLSQP',
                  constraints=cons, bounds = bounds, options={'disp': False},
                  tol=self.tolcon)
        if self.display and not res.success:
            print(res.message)
        return res.x

    def markowitz_weight_bnd_std(self, target_std):
        '''
        input: returns dataframe
        cov = returns.cov() / avg_cov(returns)
        target = targeted sharpe ratio (this is not annualized)
        (returns, cov, init_weight, bm_weight, bm_bnd, gross_weight, target_mu)
        (returns, cov, init_weight, bnd, gross_weight, target_mu)
        '''
        V = np.array(self.cov) #covariance
        R = self.returns.mean()
    
        w0 = self.init_weight

        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight},
                {'type': 'eq', 'fun': lambda x: np.sqrt(x@V@x.T)-target_std})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                    {'type': 'eq', 'fun': lambda x: np.sqrt(x@V@x.T)-target_std})

        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*self.returns.shape[1])
        
        res= minimize(return_objective, w0, args=[R], method='SLSQP',
                  constraints=cons, bounds = bounds, options={'disp': False},
                  tol=self.tolcon)
        if self.display and not res.success:
            print(res.message)
            
        return res.x  
    
    def riskparity_weight(self, risk_weight):
        '''
        input: returns dataframe
        cov = returns.cov() / avg_cov(returns)
        risk_weight = list of weights that sum up to 1
        (cov, init_weight, bm_weight, bm_bnd, gross_weight, risk_weight)
        (cov, init_weight, bnd, gross_weight, risk_weight)
        '''
        V = np.array(self.cov) #covariance
    
        w0 = self.init_weight
    
        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight})  
            
        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*len(self.init_weight))
        
        res= minimize(risk_budget_objective, w0, args=[V,risk_weight], method='SLSQP',
                  constraints=cons, bounds = bounds, options={'disp': False},
                  tol=self.tolcon)
        
        if self.display and not res.success:
            print(res.message)
            
        return res.x
    
    def ls1_gmv_weight(self):
        '''
        input returns dataframe
        returns unconstrained markowitz weight - max sharpe ratio
        net exposure of 1
        cov = returns.cov() / avg_cov(returns)
        '''
        return np.dot(np.linalg.inv(self.cov), np.ones(self.returns.shape[1]))/np.sum(np.dot(np.linalg.inv(self.cov), np.ones(self.returns.shape[1])))
        
    def gmv_weight_bnd(self):
        '''
        input:
        cov = returns.cov() / avg_cov(returns)
        target = targeted sharpe ratio (this is not annualized)
        (cov, init_weight, bm_weight, bm_bnd, gross_weight)
        (cov, init_weight, bnd, gross_weight)
        '''
        V = np.array(self.cov) #covariance
    
        w0 = self.init_weight
        
        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight})  
            
        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*len(self.init_weight))
        
        res= minimize(std_objective, w0, args=[V], method='SLSQP',
                  constraints=cons, bounds = bounds, options={'disp': False},
                  tol=self.tolcon)
        if self.display and not res.success:
            print(res.message)
        return res.x
    
    def md_weight(self):
        '''
        most diversified
        '''
        V = np.array(self.cov) #covariance
    
        w0 = self.init_weight

        if self.gross_weight != None:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight},
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-self.gross_weight})   
        else:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-self.net_weight})  
            
        if self.bm_weight: bounds = tuple([(x*self.bm_bnd[0],x*self.bm_bnd[1]) for x in self.bm_weight])
        elif self.bnd: bounds = tuple([self.bnd]*len(self.init_weight))
        
        res= minimize(diversify_objective, w0, args=[V], method='SLSQP',
                  constraints=cons, bounds = bounds, options={'disp': False},
                  tol=self.tolcon)
        if self.display and not res.success:
            print(res.message)
            
        return res.x

def monthly_backtest_weights(returns  : pd.DataFrame,
                             lookback : int,
                             opt_type : str,
                             cov_type : str = None,
                             ############################################
                             bound : tuple = None,
                             net_weight : int = None,
                             gross_weight : int = None,
                             display : bool = True,
                             ############################################
                             target_mu : float = None,
                             target_std : float = None,
                             risk_weight : np.array = None,
                             add : list = None,
                             ############################################
                             threshold : float = None,
                             pca_num : int = None,
                             factors : np.array = None,
                             intercept : bool = False):

    mnths = returns.index.strftime('%Y-%m').unique()
    weights = []
    for i in tqdm(range(lookback-1, len(mnths))):
        returns_temp = returns.loc[mnths[i-lookback+1]:mnths[i]]
        cov_temp = covariance_generator(returns_temp, cov_type, threshold, pca_num, factors, intercept)
        
        weight_generator = WeightOptimizer(returns=returns_temp, cov=cov_temp,
                                           init_weight = initial_weight(returns, 'default'),
                                           bnd=bound, net_weight=net_weight, tolcon = 1e-10, display=True)
        
        if opt_type == 'sharpe':
            w_temp = weight_generator.markowitz_weight_bnd_sr()
        elif opt_type == 'target mu':
            w_temp = weight_generator.markowitz_weight_bnd_mu(target_mu = target_mu)
        elif opt_type == 'target std':
            w_temp = weight_generator.markowitz_weight_bnd_std(target_std = target_std)
        elif opt_type == 'risk parity':
            w_temp = weight_generator.riskparity_weight(risk_weight)
        elif opt_type == 'gmv long only':
            w_temp = weight_generator.gmv_weight_bnd()
        elif opt_type == 'max diversify':
            w_temp = weight_generator.md_weight()
        elif opt_type == 'drawdown':
            w_temp = weight_generator.custom_weight(drawdown_objective)
        elif opt_type == 'skewness':
            w_temp = weight_generator.custom_weight(skewness_objective)
        elif opt_type == 'VaR':
            w_temp = weight_generator.custom_weight(varisk_objective, add)
        elif opt_type == 'CVaR':
            w_temp = weight_generator.custom_weight(cvarisk_objective, add)
        
        weights.append(w_temp)
        
    return np.array(weights)