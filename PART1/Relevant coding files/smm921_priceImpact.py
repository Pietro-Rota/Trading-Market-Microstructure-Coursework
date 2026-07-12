#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:25:19 2023

@author: richgpayne
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('./RGP_TICKTIME_EU.csv')



data['dateFormatted'] = pd.to_datetime(data['dateString'].str[:29], format="%Y-%m-%dT%H:%M:%S.%f")
data = data.drop(columns=['Domain', 'GMT Offset','ExchangeTime','BidSize','AskSize'])
data['mElapsed'] = data.dateFormatted.dt.hour*60 + data.dateFormatted.dt.minute
data = data[data.mElapsed>8*60 + 15 -1]
data = data[data.mElapsed<16.5*60 - 4]
data = data.drop(columns=['mElapsed','dateString'])
data = data.dropna(subset=['Bid', 'Ask'])
data['mid'] = 0.5*(data['Bid'] + data['Ask'])
data['spread'] = 10000*(data['Ask']-data['Bid'])/data['mid']
data = data.mask(data.spread<=0)
data['returns'] = data.mid.pct_change()*10000
data.loc[data['returns']<-20,'returns']=np.nan

plt.plot(data.mid)
plt.title("Mids")
plt.savefig('AZN_tickMids.pdf')  
plt.show()
plt.close()

plt.plot(data.spread)
plt.title("Spreads")
plt.savefig('AZN_tickSpreads.pdf')  

plt.show()
plt.close()

plt.plot(data.returns)
plt.title("Returns")
plt.savefig('AZN_tickReturns.pdf')  
plt.show()
plt.close()

"Create trade indicator variable and run regression" 

data['returns'] = data['returns'].fillna(0)

data.loc[data['Price']>=data['Ask'],'x']=+1
data.loc[data['Price']<=data['Bid'],'x']=-1
data['x'] = data['x'].fillna(0)

Xdata = data.loc[:,'x']
Xdata = sm.add_constant(Xdata)
Xdata['xl1'] = Xdata['x'].shift(1)
Xdata['xl2'] = Xdata['x'].shift(2)
Xdata['xl3'] = Xdata['x'].shift(3)
Xdata['xl4'] = Xdata['x'].shift(4)
Xdata['xl5'] = Xdata['x'].shift(5)
Xdata['xl6'] = Xdata['x'].shift(6)
Xdata['xl7'] = Xdata['x'].shift(7)
Xdata['xl8'] = Xdata['x'].shift(8)
Xdata['xl9'] = Xdata['x'].shift(9)
Xdata['xl10'] = Xdata['x'].shift(10)
Xdata = Xdata.drop(columns=['x'])


y = data.loc[:,'returns']

model = sm.OLS(y,Xdata,missing='drop')
result = model.fit(cov_type='HC0')
result.summary()

impactSeries = result.params.iloc[1:].cumsum()

plt.plot(impactSeries)
plt.title("Cumulative price impact: 10n events post-trade")
plt.savefig('AZN_priceImpact.pdf')  
plt.show()
plt.close()