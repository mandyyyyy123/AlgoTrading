# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:47 2020

@author: Mandy
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt # For debug only

def kalman(y, x, delta, Ve):
# =============================================================================
# calculation of Hurst exponent given log price series z
#
# input: 
#  y: dependent variable
#  x: independent variables
#  Vw: state variance (TODO:to be calibrated properly)
#  delta: parameter to generate the diagnal value for Vw 
#  Ve: observation/measurement variance
# output:
#  Ve
#  beta
#  Q
# =============================================================================
	n = x.shape[1]
	yhat=np.full(y.shape[0], np.nan) # measurement prediction
	e=yhat.copy()
	Q=yhat.copy()

	# For clarity, we denote R(t|t) by P(t). Initialize R, P and beta.
	R=np.zeros((n,n))
	P=R.copy()
	beta=np.full((n, x.shape[0]), np.nan)

	Vw=delta/(1-delta)*np.eye(n) # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).

	beta[:, 0]=1 # t = 0

	for t in range(len(y)):
		if t > 0:
			beta[:, t]=beta[:, t-1]
			R=P+Vw
		yhat[t]=np.dot(x[t, :], beta[:, t])
	#   print('FIRST: yhat[t]=', yhat[t])

		Q[t]=np.dot(np.dot(x[t, :], R), x[t, :].T)+Ve
		# print('Q[t]=', Q[t])
		# Observe y(t)
		e[t]=y[t]-yhat[t] # measurement prediction error
		#   print('e[t]=', e[t])
		#   print('SECOND: yhat[t]=', yhat[t])   
		
		K=np.dot(R, x[t, :].T)/Q[t] #  Kalman gain
		#    print(K)
		beta[:, t]=beta[:, t]+np.dot(K, e[t]) #  State update. Equation 3.11
		#    print(beta[:, t])
		# P=R-np.dot(np.dot(K, x[t, :]), R) # State covariance update. Euqation 3.12
		#K * X will return a matrix as P = (I - K*X) * R, hence np.outer 
		P=R-np.dot(np.outer(K, x[t, :]), R) # Thanks to Matthias for chaning np.dot -> np.outer!

	return beta, e, Q
