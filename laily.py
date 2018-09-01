# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 00:02:44 2018

@author: WestHamster
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy as np
import pandas as pd

sets = []
Xtrain = []
Ytrain =[]


ds = open("lily.txt", 'r')
for line in ds:
    if len(line.split()) != 0:
        sets.append(line.split())
for items in sets:
    Xtrain.append(items[0:4])
    Ytrain.append(items[5:])
Xtrain = np.array(Xtrain)
digits = Xtrain
train = Ytrain
print(digits)
clf = svm.SVC(gamma=0.001, C=100)
print(clf)
"""
for item in sets:
    Xtrain.append(item[0:4])
    Ytrain.append(item[5:])

#Xtrain = np.array(Xtrain)
#Ytrain = np.array(Ytrain)

print(Xtrain)
print(Ytrain)
"""
