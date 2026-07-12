#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:25:19 2023

@author: richgpayne
"""

myStock = "AZN.L"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./RGP_Europe_60s.csv')
data['dateFormatted'] = pd.to_datetime(data['dateString'].str[:26], format="%Y-%m-%dT%H:%M:%S.%f")
data = data.drop(columns=['Domain', 'Type'])
data['mElapsed'] = data.dateFormatted.dt.hour*60 + data.dateFormatted.dt.minute
data = data[data.mElapsed>8*60 + 15 -1]
data = data[data.mElapsed<16.5*60 - 4]
data = data.drop(columns=['mElapsed','dateString'])
data['mid'] = 0.5*(data['Bid'] + data['Ask'])
data['spread'] = 10000*(data['Ask']-data['Bid'])/data['mid']
data['depth'] = 0.5*(data['AskSize']+data['BidSize'])
data = data.mask(data.spread<=0)

data=data[data.Stock==myStock]

adv = data.Volume.sum()/len(data.dateFormatted.dt.date.unique())
adt = data.Trades.sum()/len(data.dateFormatted.dt.date.unique())

plt.plot(data.mid)
plt.title(myStock + " Mids")
plt.savefig(myStock[0:3] + '_mid.pdf')  
plt.show()
plt.close()

plt.plot(data.spread)
plt.title(myStock + " Spreads")
plt.savefig(myStock[0:3] + '_spread.pdf')  
plt.show()
plt.close()

plt.plot(data.depth)
plt.title(myStock + " Depth")
plt.savefig(myStock[0:3] + '_depth.pdf')  
plt.show()
plt.close()

plt.plot(data.Volume)
plt.title(myStock + " Volume")
plt.savefig(myStock[0:3] + '_volume.pdf')  
plt.show()
plt.close()

plt.plot(data.Trades)
plt.title(myStock + " Trades")
plt.savefig(myStock[0:3] + '_trades.pdf')  
plt.show()
plt.close()

hourlyMeans = data.groupby([data["dateFormatted"].dt.hour]).mean()

plt.plot(hourlyMeans.spread)
plt.title(myStock + " Hourly Mean Spreads")
plt.savefig(myStock[0:3] + '_hourlySpread.pdf')  
plt.show()
plt.close()

plt.plot(100*hourlyMeans.depth/adv)
plt.title(myStock + " Hourly Mean Depth as percentage of ADV")
plt.savefig(myStock[0:3] + '_hourlyDepth.pdf')  
plt.show()
plt.close()


plt.plot(100*hourlyMeans.Volume*60/adv)
plt.title(myStock + " Hourly Mean Volume as percentage of ADV")
plt.savefig(myStock[0:3] + '_hourlyVolume.pdf')  
plt.show()
plt.close()

plt.plot(100*hourlyMeans.Trades*60/adt)
plt.title(myStock + " Hourly Mean Trades as percentage of ADT")
plt.savefig(myStock[0:3] + '_hourlyTrades.pdf')  
plt.show()
plt.close()
