# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn import svm
import statsmodels.api as sm
import pywt
import copy
import warnings
from statsmodels.tsa.arima_model import ARMA
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as GBR
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def WT(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
    
    '''
    WT: Wavelet Transformation Function

    index_list: Input Sequence;
   
    lv: Decomposing Level；
 
    wavefunc: Function of Wavelet, 'db4' default；
    
    m, n: Level of Threshold Processing
   
    '''
   
    # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function 

    # Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold

    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
        
    if plot:     
        denoised_index = np.sum(coeff, axis=0)   
        data = pd.DataFrame({'CLOSE': index_list, 'denoised': denoised_index})
        data.plot(figsize=(10,10),subplots=(2,1))
        data.plot(figsize=(10,5))
   
    return coeff





def AR_MA(coeff):
    
    '''
    AR_MA:  Autoregressive Moving Average Function
    
    coeff:  Input sequence disposed by WT (Wavelet Transformation Function)
    
    '''
    
    warnings.filterwarnings('ignore')
    order, model, results = [], [], []

    for i in range(1, len(coeff)):
        order.append(sm.tsa.arma_order_select_ic(coeff[i], ic='aic')['aic_min_order'])   # Select (p, q) by AIC criterion 
        model.append(ARMA(coeff[i], order=order[i-1]))
    
    for i in range(len(model)):
        new_order = list(order[i])
        while True:
            try:
                results.append(model[i].fit())
                
            except ValueError:                                              # Further determinte the appropriate (p, q) for the model
                new_order[1] = np.max((0, new_order[1]-1))
                model[i] = ARMA(coeff[i+1], order=new_order)         

            if len(results)>= i+1:
                break                
    
    return results




def NonlinReg(coeff, regressor='GBR', features=4, interval=0, length=1):
    
    '''
    NonlinReg: Non-linear Regression Model
    
    coeff: Input sequence disposed by WT (Wavelet Transformation Function)
    
    regressor: Non-linear regressor, 'GBR' default
    
    features: Days used to predict, 4 default
    
    interval: Prediction lagging, 0 default
    
    length: 1 default
    '''
    X, Y = [], []
    for i in range(len(coeff[0])):
        if i+features+interval < len(coeff[0]):
            X.append(coeff[0][i:i+features])
            Y.append(coeff[0][i+features+interval])
    X =  np.array(X)
    Y =  np.array(Y)

    if regressor == 'GBR':
        gbr = GBR(learning_rate=0.1, n_estimators=80, max_depth=2).fit(X, Y)
        
        X_ = copy.deepcopy(X)
        Y_ = copy.deepcopy(Y)
        for i in range(length):
            X_ = np.concatenate((X_, np.array([np.concatenate((X_[-1][-features+1:], Y_[[-interval-1]]))])))
            Y_ = np.concatenate((Y_, gbr.predict(X_[-1])))
    
    if regressor == 'SVR':
        svr = svm.SVR(kernel='rbf', C=100, gamma=3).fit(X, Y)

        X_ = copy.deepcopy(X)
        Y_ = copy.deepcopy(Y)
        for i in range(length):
            X_ = np.concatenate((X_, np.array([np.concatenate((X_[-1][-features+1:], Y_[[-interval-1]]))])))
            Y_ = np.concatenate((Y_, svr.predict(X_[-1])))
    
    return Y_




def ModelEvaluation(index_predict, index_real, model_name='model'): 
    
    '''
    ModelEvaluation: The function used to evaluate prediction model
    
    index_predict: The predict sequence
    
    index_real: The actual sequence 
    
    modelname: For the displaying convenience, the name of index
    
    '''
    
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # The evaluation model list
    model_metrics_list = []  # The evaluation indicatiors list
    
    
    for m in model_metrics_name:  
        tmp_score = m(index_predict, index_real)  # compute each result
        model_metrics_list.append(tmp_score)
    df = pd.DataFrame(np.array([model_metrics_list]), index=[model_name], columns=['ev', 'mae', 'mse', 'r2'])
    
    
    return df
