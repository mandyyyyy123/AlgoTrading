# Kalman Filter Mean Reversion Strategy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
import statsmodels.tsa.stattools as ts
#import statsmodels.tsa.vector_ar.vecm as vm
from os import path
from kalman import kalman
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Data"))

df=pd.read_csv(filepath+'/inputData_EWA_EWC_IGE.csv')
df['Date']=pd.to_datetime(df['Date'],  format='%Y%m%d').dt.date # remove HH:MM:SS
df.set_index('Date', inplace=True)

numStates = df.shape[1] # column numbers, plus offset 
x = np.array(ts.add_constant(df.values[:,:-1])) # Augment x with ones in front to  accomodate possible offset in the multi regression between y vs x
y=np.array(df.values[:,-1]) 
print(x[:20])
#x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.

delta=0.0000007 
Ve=0.001

beta, e, Q = kalman(y,x,delta,Ve)

plt.plot(beta[0, :], label='mu')
plt.plot(beta[1, :], label='ratio1')
plt.plot(beta[2, :], label='ratio2')
plt.plot(e[4:], label='deviation')
plt.plot(np.sqrt(Q[4:]), label='std')
plt.legend()

longsEntry=e < -np.sqrt(Q)
longsExit =e > 0

shortsEntry=e > np.sqrt(Q)
shortsExit =e < 0

#numUnitsLong=np.zeros(longsEntry.shape)
#numUnitsLong[:]=np.nan
numUnitsLong = np.full(longsEntry.shape, np.nan)

numUnitsShort=np.full(shortsEntry.shape, np.nan)

numUnitsLong[0]=0
numUnitsLong[longsEntry]=1
numUnitsLong[longsExit]=0
numUnitsLong=pd.DataFrame(numUnitsLong)
numUnitsLong.fillna(method='ffill', inplace=True)

numUnitsShort[0]=0
numUnitsShort[shortsEntry]=-1
numUnitsShort[shortsExit]=0
numUnitsShort=pd.DataFrame(numUnitsShort)
numUnitsShort.fillna(method='ffill', inplace=True)
print(numUnitsShort[:10])

numUnits=numUnitsLong+numUnitsShort
alloc = np.zeros((beta.shape[0], beta.shape[1]))
alloc[:-1,:] = -beta[1:,:]
alloc[-1,:] = 1

positions=pd.DataFrame(np.tile(numUnits.values, [1, numStates]) * alloc.T*df.values) #  [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
(np.cumprod(1+ret)-1).plot(legend=True)
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
#APR=0.313225 Sharpe=3.464060
plt.show()