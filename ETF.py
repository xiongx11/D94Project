#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:58:06 2018

@author: xiaolux
"""
import math
import quantiacsToolbox
import statsmodels 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from statsmodels.tsa.stattools import coint
from datetime import datetime
from dateutil.parser import parse



# try to do for ETF "SPY" against "IVV"

def head(data):
    print(data[:10])
    
def ass(npdata):
    return np.asscalar(npdata)

def list_time(Date):
    date = []
    for i in range (0, len(Date)):       
        date.append(datetime.strptime(Date[i][0], "%Y-%m-%d"))
    return date

def py_datetime(date):
    datetime_object = date.strptime('2017-09-06', '%b %d %Y %I:%M%p')
    return datetime_object
        
    

if __name__ == '__main__':
# start data is 2013-09-13 end data is 2018-09-17
    IVVdata = pd.read_csv('IVV.csv', delimiter = ',')
    #P1 = IVVdata.loc[:,['Date','Close']]
    P1 = IVVdata.loc[:,['Close']]
    lP1 = np.log(P1['Close'])
    P1 = P1.values
    
    
    
    SPYdata = pd.read_csv('SPY.csv', delimiter = ',')
    P2 = SPYdata.loc[:,['Close']]
    lP2 = np.log(P2['Close'])
    P2 = P2.values
    logRat = lP1 - lP2 # log-ratio of prices
    stdlogRat = (logRat - np.mean(logRat))/np.std(logRat)
    
    open_lim=2.0
    bail_lim=3.5
    trade_ind=0 
    
    t_open= []
    t_close=[]
    # trade indicator: 0 means no position, 
    #1 means buy 1st & sell 2nd stock, 
    #-1 means sell 1st & buy 2nd stock
    n=len(logRat)
    profit= np.zeros((1, n))
    
    #n1 =0
    #n2 = 0
    
    
    for i in range(1,n):
      # check for closing an open position
      if((trade_ind*ass(stdlogRat[i]) <0) | (abs(ass(stdlogRat[i]))> bail_lim) ):
        trade_ind=0
        profit[0][i]=profit[0][i]+n1*P1[i]+n2*P2[i]
        t_close.append((t_close,i))
      
      # check for opening a position
      if( trade_ind == 0 and abs(ass(stdlogRat[i])) > open_lim ):
        trade_ind= ass(np.sign(stdlogRat[i]))
        n1=-trade_ind / (P1[i])
        n2=trade_ind / (P2[i])
        t_open.append((t_open, i))
        

    
    Date = IVVdata.loc[:,['Date']].values
    list_date = list_time(Date)

    
    #dates = plt.dates.date2num(list_date)
    #plt.pyplot.plot_date(dates, profit)
    
    plt.plot(list_date,profit[0])
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.show()
    #profit1 = profit.reshape(1259,1)
    
    #plt.plot(Date, profit)
    
    
    
    
    
