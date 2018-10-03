#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:18:58 2018
This function find the pairs of stocks that there difference is stationary
and ordered the pairs from most significant to not significant
Got the top 5

@author: xiaolux
"""
import math
import qtb
import statsmodels 
import numpy as np
import pandas as pd
from pandas import Series;
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import seaborn
import statsmodels.api as sm
from sklearn import linear_model


from statsmodels.tsa.stattools import coint, adfuller, kpss
from datetime import datetime
from dateutil.parser import parse

def test_stationarity(timeseries):
    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(24).mean()
    rolstd = timeseries.rolling(24).std()
    #rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], O=['Test Statistic',
                         'p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
def test_stationary_pval(timeseries):
    dftest0 = adfuller(timeseries, autolag='AIC')
    # if pvalue small than accpet?
    return dftest0[1]

def fine_stationary_pair(filenames, year1, year2):
    # the filenames passed in here is the stock name in my file
    pvals_small = []
    pvals_small_pair = []
    n = len(filenames)
    
    i = 0
    while i < n:
        j = 0
        while j < n:
            if i < j:
                its = write_csv_to_ts(filenames[i])
                
                jts = write_csv_to_ts(filenames[j])
                
                diff  = its - jts
                diff = diff[year1:year2]
                #print [i,j]
                #print test_stationary_pval(diff)
                if test_stationary_pval(diff) <=0.16:
                   pvals_small.append((i,j)) 
                   pvals_small_pair.append((filenames[i], filenames[j]))
            j = j+1
        i = i+1
               
    return pvals_small, pvals_small_pair

def get_pairs(filename, year1, year2):
    filenames =  read_filenames('filenames.txt')
    
    stationary_pair = fine_stationary_pair(filenames, year1, year2)
    
    return stationary_pair 

def adf_test(y):
    # perform Augmented Dickey Fuller test
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    return(dfoutput[1])
    
#  find adf p values and the pairs    
def find_adf_pvalues(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            # LOG OF PRICE
            price = pd.concat([S1, S2], axis=1)
            price.columns = [keys[i],keys[j]] 
            lp = np.log(price)
            x = lp[keys[i]]
            y = lp[keys[j]]
            spread = reg(x,y)

            # check if the spread is stationary 
            adf = sm.tsa.stattools.adfuller(spread, maxlag=1)
            pvalue_matrix[i, j] = adf[1]
            if adf[1] < 0.01:
                pairs.append((adf[1],keys[i], keys[j]))

    return pairs
     


def significance(pairs):
    ordered_pair = []
    k = 0
    small = 1           
    while k < len(pairs):
        if pairs[k][2] < small:
            small = pairs[k][2]
            ordered_pair.append(pairs[k])
        k = k+1
    return ordered_pair


                

def reg(x,y):
    # fit the data to linear regression and return spread
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x,pd.Series([1]*len(x),index = x.index)], axis=1)
    regr.fit(x_constant, y)    
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x*beta - alpha
    return spread




if __name__ == '__main__':
#    
    Markets=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
         'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
     'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
     'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
     'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
     'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
     'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
     'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']
#    Markets=['ABT','ORCL','ACN','JPM','ALL','MDT','AMGN','FDX','APC',
#             'HAL']

    
    #print get_pairs('filenames.txt', '2009', '2018')
    data = qtb.loadData(Markets, qtb.REQUIRED_DATA,False,'20120506','20150506')  
    closeData = pd.DataFrame(data['CLOSE'],columns=Markets)

    closeData = closeData.dropna(axis='columns')
    

    pairs =  find_adf_pvalues(closeData)
    pairs.sort()
    print pairs
    
#    ordered_pair = []
#    k = 0
#    small = 1           
#    for items in pairs:
#        if items[k][2] < small:
#            small = items[k][2]
#            ordered_pair.append(items[k])
#        k = k+1
#    print ordered_pairs
    
#    find_pairs(Markets,closeData)

    #print reg()
