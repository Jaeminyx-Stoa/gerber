# -*- coding: utf-8 -*-

import numpy as np

###################
# basic functions #
###################

def risk_contribution(w,V):
    '''
    returns risk contribution of each asset which sums up to total std
    '''
    V = np.array(V)
    w = np.array(w)
    sigma = np.sqrt(w@V@w.T)
    # Marginal Risk Contribution
    MRC = V@w.T

    # marginal contribution
    RC = (MRC * w)/sigma
    
    return RC #this sums up to sigma

#####################################
# objectives: needs to be minimized #
#####################################

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(x@V@x.T)# portfolio standard deviation
    
    risk_target = sig_p * x_t
    current_RC = risk_contribution(x,V)

    J = sum(np.square(current_RC-risk_target)) * 5000 # sum of squared error
    return J

def sharpe_objective(x, pars):
    R = pars[0]
    V = pars[1]
    sig_p = np.sqrt(x@V@x.T)
    r_p = R @ x
    sharpe_ratio = r_p / sig_p
    J = -sharpe_ratio
    return J

def std_objective(x, pars):
    V = pars[0]
    
    sig_p = np.sqrt(x@V@x.T) #portfolio standard deviation
    return sig_p

def return_objective(x, pars):
    R = pars[0]
    r_p = R @ x
    return -r_p

def diversify_objective(x, pars):
    V = pars[0]
    
    sig_p = np.sqrt(x@V@x.T)/(np.diag(V)*x).sum() #portfolio standard deviation
    return sig_p

def drawdown_objective(x, pars):
    returns = pars[0]
    pf_r = returns @ x
    return -min(np.cumprod(pf_r+1)/np.maximum.accumulate(np.cumprod(pf_r+1))-1)
    
def skewness_objective(x, pars):
    returns = pars[0]
    pf_r = returns @ x
    return -pf_r.skew()

    
def varisk_objective(x, pars):
    returns = pars[0]
    pf_r = returns @ x
    percentile = pars[1] # ex. 0.05
    
    return -pf_r.quantile(percentile)

def cvarisk_objective(x, pars):
    returns = pars[0]
    pf_r = returns @ x
    percentile = pars[1] # ex. 0.05
    
    return - pf_r[pf_r<=pf_r.quantile(percentile)].mean()
