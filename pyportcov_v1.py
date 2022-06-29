# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def nondiag_avg(matrix2d : np.array) -> np.array:
    num = matrix2d.shape[0]
    return (matrix2d.sum() - np.diag(matrix2d).sum()) / (num*(num-1))

def avg_corr(returns : np.array) -> tuple:
    ss = np.array(returns.std()).reshape(-1, 1) @ np.array(returns.std()).reshape(1, -1)
    corr = np.array(returns.corr())
    num = returns.shape[1]
    
    nondiag = (np.ones((num, num)) - np.eye(num)) * nondiag_avg( corr )
    diag = np.eye(num) * corr
    
    return (nondiag + diag), ss * (nondiag + diag)
    
def avg_cov(returns : np.array) -> tuple:
    ss = np.array(returns.std()).reshape(-1, 1) @ np.array(returns.std()).reshape(1, -1)
    cov = np.array(returns.cov())
    num = returns.shape[1]
    
    nondiag = (np.ones((num, num)) - np.eye(num)) * nondiag_avg( cov )
    diag = np.eye(num) * cov
    
    return (nondiag + diag)/ss, nondiag + diag
    
def gerber_cov_stat0(rets : np.array,
                     threshold : float = 0.5) -> tuple:
    """
    compute Gerber covariance Statistics 0, orginal Gerber statistics, not always PSD
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 > threshold > 0, "threshold shall between 0 and 1"
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                    
            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (pos + neg)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return cor_mat, cov_mat

def gerber_cov_stat1(rets : np.array,
                     threshold : float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 1
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 > threshold > 0, "threshold shall between 0 and 1"
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    #sd_vec = np.ones(p)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return cor_mat, cov_mat

def gerber_cov_stat2(rets : np.array,
                     threshold : float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 2
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    U = np.copy(rets)
    D = np.copy(rets)

    # update U and D matrix
    for i in range(p):
        U[:, i] = U[:, i] >= sd_vec[i] * threshold
        D[:, i] = D[:, i] <= -sd_vec[i] * threshold
        
    # update concordant matrix
    N_CONC = U.transpose() @ U + D.transpose() @ D

    # update discordant matrix
    N_DISC = U.transpose() @ D + D.transpose() @ U
    H = N_CONC - N_DISC
    h = np.sqrt(H.diagonal())

    # reshape vector h and sd_vec into matrix
    h = h.reshape((p, 1))
    sd_vec = sd_vec.reshape((p, 1))

    cor_mat = H / (h @ h.transpose())
    cov_mat = cor_mat * (sd_vec @ sd_vec.transpose())
    return cor_mat, cov_mat #, U, D

def gerber_cov_stat3(rets : np.array,
                     threshold : float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 2
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    U = np.copy(rets)
    D = np.copy(rets)

    # update U and D matrix
    for i in range(p):
        U[:, i] = U[:, i] >= sd_vec[i] * threshold
        U[:, i][rets[:, i] >= sd_vec[i] * threshold * 2] = np.sqrt(2) # additional bucket
        
        D[:, i] = D[:, i] <= -sd_vec[i] * threshold
        D[:, i][rets[:, i] <= -sd_vec[i] * threshold * 2] = np.sqrt(2) # additional bucket
        
    # update concordant matrix
    N_CONC = U.transpose() @ U + D.transpose() @ D

    # update discordant matrix
    N_DISC = U.transpose() @ D + D.transpose() @ U
    H = N_CONC - N_DISC
    h = np.sqrt(H.diagonal())

    # reshape vector h and sd_vec into matrix
    h = h.reshape((p, 1))
    sd_vec = sd_vec.reshape((p, 1))

    cor_mat = H / (h @ h.transpose())
    cov_mat = cor_mat * (sd_vec @ sd_vec.transpose())
    return cor_mat, cov_mat #, U, D

def pca_cov(returns : np.array,
            pca_num : int = None,
            threshold : float=None) -> tuple:
    evals, evecs = np.linalg.eig(returns.cov())
    sidx = np.argsort(evals)[::-1]
    evals = evals[sidx]
    evecs = evecs[:, sidx]
    
    ss = np.array(returns.std()).reshape(-1, 1) @ np.array(returns.std()).reshape(1, -1)
    if pca_num:
        cov = (evecs[:,:pca_num] @ np.diag(evals[:pca_num]) @ evecs[:,:pca_num].T)
        corr = cov/ss
    elif threshold:
        n = np.where(np.cumsum(evals/sum(evals))>threshold)[0][0]
        cov = (evecs[:,:n+1] @ np.diag(evals[:n+1]) @ evecs[:,:n+1].T)
        corr = cov/ss
    return corr, cov

def factor_cov(returns : np.array,
               factors : np.array,
               intercept : bool = False) -> tuple:
    
        ss = np.array(returns.std()).reshape(-1, 1) @ np.array(returns.std()).reshape(1, -1)
        
        if intercept:
            X = np.concatenate([np.ones([len(factors),1]),factors], axis=1)
        else:
            X = factors * 1
            
        betas = np.linalg.solve(X.T@X, X.T@returns)
        # X @ betas
        
        if factors.shape[1] == 1:
            cov = betas.T @ np.cov(X.T).reshape(1,1) @ betas
        else:
            cov = betas.T @ np.cov(X.T) @ betas
        corr = cov/ss
        
        return corr, cov
    
def covariance_generator(returns : np.array,
                         cov_type : str,
                         threshold : float = None,
                         pca_num : int = None,
                         factors : np.array = None,
                         intercept : bool = False) -> tuple:
    """
    cov_type: ['sample','avg_corr','avg_cov','gerber0','gerber1','gerber2','gerber3','pca','mfactor']
    """
    
    if cov_type == "sample":
        return returns.cov()
    elif cov_type == "avg_corr":
        return avg_corr(returns)[1]
    elif cov_type == "avg_cov":
        return avg_cov(returns)[1]
    elif cov_type == "gerber0":
        return gerber_cov_stat0(np.array(returns), threshold)[1]
    elif cov_type == "gerber1":
        return gerber_cov_stat1(np.array(returns), threshold)[1]
    elif cov_type == "gerber2":
        return gerber_cov_stat2(np.array(returns), threshold)[1]
    elif cov_type == "gerber3":
        return gerber_cov_stat3(np.array(returns), threshold)[1]
    elif cov_type == "pca":
        return pca_cov(returns, pca_num, threshold)[1]
    elif cov_type == "mfactor":
        return factor_cov(returns, factors, intercept)[1]