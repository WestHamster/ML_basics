# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:23:55 2018

@author: WestHamster
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy

digits =datasets.load_digits()
print(digits.data)
print(digits.target)
print(digits.images[0])
print()
print()
clf=svm.SVC(gamma=0.1,C=100)
print(len(digits.data))
x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print("Prediction data:",clf.predict(digits.data[29]))
plt.imshow(digits.images[9],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()
