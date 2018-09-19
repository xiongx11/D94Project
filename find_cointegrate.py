#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:18:28 2018

@author: xiaolux

This is a file trying to find integrated pairs on commodities markets
"""
import qtb
import statsmodels 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

#def get_graph(stock1, stock2):
#    S1 = closeData[stock1]
#    S2 = closeData[stock2]
#    ratios = S1 / S2
#
#    trainingnumber = len(ratios)*2//3
#    train = ratios
#    test = ratios[trainingnumber:]
#    ratios_mavg5 = train.rolling(window=5,
#                               center=False).mean()
#    ratios_mavg60 = train.rolling(window=60,
#                               center=False).mean()
#    std_60 = train.rolling(window=60,
#                        center=False).std()
#    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
#    plt.figure(figsize=(15,7))
#    plt.plot(train.index, train.values)
#    plt.plot(ratios_mavg5.index, ratios_mavg5.values)
#    plt.plot(ratios_mavg60.index, ratios_mavg60.values)
#    plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
#    plt.ylabel('Ratio')
#    plt.show()
#    plt.figure(figsize=(15,7))
#    zscore_60_5.plot()
#    plt.axhline(0, color='black')
#    plt.axhline(1.0, color='red', linestyle='--')
#    plt.axhline(-1.0, color='green', linestyle='--')
#    plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
#    plt.show()
    
def split_data(stock1, stock2):
    ratios = closeData[stock1] / closeData[stock2]
    n = len(ratios)
    train = ratios[:(n*2/3)]
    test = ratios[(n/1/3):]
    return ratios, train, test

def get_feature(train):
    """
    60 day Moving Average of Ratio: Measure of rolling mean
    5 day Moving Average of Ratio: Measure of current value of mean
    60 day Standard Deviation
    z score: (5d MA — 60d MA) /60d SD """
    ratios_mavg5 = train.rolling(window=5,
                               center=False).mean()
    ratios_mavg60 = train.rolling(window=60,
                               center=False).mean()
    std_60 = train.rolling(window=60,
                        center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    
    return ratios_mavg5, ratios_mavg60, std_60, zscore_60_5

def plot_feature(ratios_mavg5, ratios_mavg60, std_60, zscore_60_5):
    
    plt.figure(figsize=(15,7))
    plt.plot(train.index, train.values)
    plt.plot(ratios_mavg5.index, ratios_mavg5.values)
    plt.plot(ratios_mavg60.index, ratios_mavg60.values)
    plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
    plt.ylabel('Ratio')
    plt.show()

    plt.figure(figsize=(15,7))
    zscore_60_5.plot()
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
    plt.show()

def plot_ratios(ratios,train, test,zscore_60_5):
    ''' Plot the ratios and buy and sell signals from z score '''
    plt.figure(figsize=(15,7))
    train[60:].plot()
    buy = train.copy()
    sell = train.copy()
    buy[zscore_60_5>-1] = 0
    sell[zscore_60_5<1] = 0
    buy[60:].plot(color='g', linestyle='None', marker='^')
    sell[60:].plot(color='r', linestyle='None', marker='^')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,ratios.min(),ratios.max()))
    plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
    plt.show()
    # it seems that we are mostly selling it and rarely buy it ,I am not sure 
    # is it because the stock ratio is mostly going down or I am having a bad 
    # strategy
    
    # then plot the testing buy and sell signals 

def plot_testing(closeData, stock1, stock2, ratios,train, test):
    """ plotted both hypothetical and testing"""
    plt.figure(figsize=(18,9))
    n = len(ratios)
    S1 = closeData[stock1].iloc[:(n/2)]
    S2 = closeData[stock2].iloc[:(n/2)]
    S1[60:].plot(color='b')
    S2[60:].plot(color='c')
    
    buyR = 0*S1.copy()
    sellR = 0*S1.copy()
    buy = train.copy()
    sell = train.copy()
    # When buying the ratio, buy S1 and sell S2
    buyR[buy!=0] = S1[buy!=0]
    sellR[buy!=0] = S2[buy!=0]
    # When selling the ratio, sell S1 and buy S2 
    buyR[sell!=0] = S2[sell!=0]
    sellR[sell!=0] = S1[sell!=0]
    buyR[60:].plot(color='g', linestyle='None', marker='^')
    sellR[60:].plot(color='r', linestyle='None', marker='^')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,min(S1.min(),S2.min()),max(S1.max(),S2.max())))
    plt.legend([stock1, stock2, 'Buy Signal', 'Sell Signal'])
    plt.show()
    
# Trade using a simple strategy
def trade(S1, S2, window1, window2):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] > 1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < 1
        elif zscore[i] < -1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
            
            
    return money

    
    
if __name__ == '__main__':
    #remove first 4 for some how I get all nans
    commoditiesMarkets = ["ATI", "CF", "FMC",
                          "IFF", "LYB", "MOS", "NUE", "PPG", "SHW"]
    
    data = qtb.loadData(commoditiesMarkets, qtb.REQUIRED_DATA,False,'20120506','20150506')
    closeData = pd.DataFrame(data['CLOSE'],columns=commoditiesMarkets)
    
    
    # run for utility Martkets
        
    #utilityMarkets = ["AES","AEE", "AEP","CNP","CMS","ED","D","DTE","DUK",
                   #   "EIX","ETR","EXC","FE","NEE","NI","NRG","PCG","PSX",
                    #  "PNW","PPL","PEG","SCG","SRE","SO","WEC"]
    
    #data = qtb.loadData(utilityMarkets, qtb.REQUIRED_DATA,False,'20120506','20150506')
    #closeData = pd.DataFrame(data['CLOSE'],columns=utilityMarkets)
    
    

    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(closeData)
    #print(pairs)
    stock1 = pairs[0][0]
    stock2 = pairs[0][1]
    
    #try this "SPY" against "IVV"
    #etf = ["SPY", "IVV"]
    #dataEtf = qtb.loadData(etf, qtb.REQUIRED_DATA,False,'20120506','20150506')
    
    
    ratios, train, test = split_data(stock1, stock2)
    ratios_mavg5, ratios_mavg60, std_60, zscore_60_5 = get_feature(train)
    #plot_feature(ratios_mavg5, ratios_mavg60, std_60, zscore_60_5)
    
    #plot_ratios(ratios, train, test,zscore_60_5)
    #plot_testing(closeData, stock1, stock2, ratios,train, test)
    
    S1 = closeData[stock1].iloc[:1763]
    S2 = closeData[stock2].iloc[:1763]
    trade(S1, S2, 60, 5)
    
    # St/Pt
# 1, look at log ratio log() 
# 2, calculate profit adjustment for interest
    #mean reverting probability # trasaction cost few transcation less 
    

    
    