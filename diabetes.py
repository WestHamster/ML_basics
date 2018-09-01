# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:25:58 2018

@author: WestHamster
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score


datam = pd.read_csv("C:/Users/WestHamster/Downloads/DATASET/diabetes.csv")
datam.head()
datas = datasets.load_diabetes()
dataX = datas.data[:, np.newaxis, 2]

print(datas)
trainX = dataX[:-600]
testX = dataX[-600:]
trainY = print(datas.target[:-600])
testY = print(datas.target[-600:])

regr = linear_model.LinearRegression()
regr.fit(trainX,trainY)

pred = regr.predict(testX)
print("Coeff:",regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(testY, pred))
print('Variance score: %.2f' % r2_score(testY, pred))

