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
from pandas import Series;
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import seaborn

from statsmodels.tsa.stattools import coint, adfuller, kpss
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

def write_csv_to_ts(filename):

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    data = pd.read_csv(filename, parse_dates=['Date'], 
                       index_col='Date',date_parser=dateparse)
    ts = data['Close']
    return(ts)

def get_time_period(ts, year1, year2):
    #ts[year1:]
    ts_new = ts[year1:year2]
    return ts_new



    
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
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                         'p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

def test_stationary_pval(timeseries):
    dftest0 = adfuller(timeseries, autolag='AIC')
    # if pvalue small than accpet?
    return dftest0[1]

def read_filenames(filename):
    # the filename passed here is a file that contains all the names of stocks
    with open (filename, "r") as myfile:
        tlines = [line.rstrip('\n') for line in myfile]
    return tlines

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

def my_handle_data(context, data):
    """
    Called every day.
    """
    If get_open_orders():
        return
    

if __name__ == '__main__':
    IVVts = write_csv_to_ts('IVV.csv')
    SPYts = write_csv_to_ts('SPY.csv')
    diff_ts = IVVts - SPYts
    #test_stationarity(diff_ts)
    newdiff = get_time_period(diff_ts, '2009', '2018')
     
#    filenames =  read_filenames('filenames.txt')
#    stationary_pair = fine_stationary_pair(filenames)
#    print stationary_pair
#    s1_index = stationary_pair[0][0]
#    s1 =  filenames[s1_index]
#    s2_index = stationary_pair[0][1]
#    s2 = filenames[s2_index]
#    print s1,s2
#    
    
    
    # try to read all pairs and trade 3 pairs
    print get_pairs('filenames.txt', '2009', '2018')
    # get all pairs names
    
    
    
    

            
       # write_csv_to_ts(name)
    
    
#    plt.plot(IVVts)
#    plt.plot(SPYts)
#    plt.plot(diff_ts)
    #test_stationarity(diff_ts)
#    print test_stationary_pval(newdiff)

# this is the p values
    
    
#if __name__ == "__main__":   
# test_stationarity(ts)
# start data is 2013-09-13 end data is 2018-09-17
#    headers = ['Date','Open','High','Low','Close','AdjClose','Volume']
#    dtypes = {'Date': 'str', 'Open':'float','High':'float',
#              'Low':'float','Close':'float','AdhClose':'float',
#              'Volume':'int' }

#    data_parser = pd.to_datetime
#    
#    IVVdata = pd.read_csv('IVV.csv', delimiter = ',', dtype = dtypes,
#                          parse_dates = parse_dates)
#    
#    IVVdata = pd.read_csv('IVV.csv')
#    
##    print(IVVdata.dtypes)
#    
#    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#    IVVdata = pd.read_csv('IVV.csv', parse_dates=['Date'], 
#                       index_col='Date',date_parser=dateparse)
##    print data.head()
##    print(data.dtypes)
##    print(data.index)
#    ts = IVVdata['Close']
#    #2. Import the datetime library and use 'datetime' function:
#
#    print ts['2013-09-18']
#    print ts['2013']
##    print ts.head()
#    plt.plot(ts)

#    
    
    
    
   
    
    
    
    
# print(series['Close'][0:10])
    #print(series['Date'])
    #plt.pyplot.plot_date(series['Date'], series['Close'])
    #plt.plot(x =series['Date'],  y = series['Close'])


    #print(type(series))
#    print(series)
#    series.plot()
#    plt.pyplot.show()
#    
#    P1 = IVVdata.loc[:,['Close']]
#    lP1 = np.log(P1['Close'])
#    P1 = P1.values
#    
    
#    
#    SPYdata = pd.read_csv('SPY.csv', delimiter = ',')
#    P2 = SPYdata.loc[:,['Close']]
#    lP2 = np.log(P2['Close'])
#    P2 = P2.values
#    logRat = lP1 - lP2 # log-ratio of prices
#    stdlogRat = (logRat - np.mean(logRat))/np.std(logRat)
#    
#    open_lim=2.0
#    bail_lim=3.5
#    trade_ind=0 
#    
#    t_open= []
#    t_close=[]
#    # trade indicator: 0 means no position, 
#    #1 means buy 1st & sell 2nd stock, 
#    #-1 means sell 1st & buy 2nd stock
#    n=len(logRat)
#    profit= np.zeros((1, n))
#    
#    #n1 =0
#    #n2 = 0
#    
#    
#    for i in range(1,n):
#      # check for closing an open position
#      if((trade_ind*ass(stdlogRat[i]) <0) | (abs(ass(stdlogRat[i]))> bail_lim) ):
#        trade_ind=0
#        profit[0][i]=profit[0][i]+n1*P1[i]+n2*P2[i]
#        t_close.append((t_close,i))
#      
#      # check for opening a position
#      if( trade_ind == 0 and abs(ass(stdlogRat[i])) > open_lim ):
#        trade_ind= ass(np.sign(stdlogRat[i]))
#        n1=-trade_ind / (P1[i])
#        n2=trade_ind / (P2[i])
#        t_open.append((t_open, i))
#        
#
#    
#    Date = IVVdata.loc[:,['Date']].values
#    list_date = list_time(Date)
#
#    
#    #dates = plt.dates.date2num(list_date)
#    #plt.pyplot.plot_date(dates, profit)
#    
#    plt.plot(list_date,profit[0])
#    # beautify the x-labels
#    plt.gcf().autofmt_xdate()
#    plt.show()
#    #profit1 = profit.reshape(1259,1)
#    
#    #plt.plot(Date, profit)
#    
#    
#    
#    
#    
    


