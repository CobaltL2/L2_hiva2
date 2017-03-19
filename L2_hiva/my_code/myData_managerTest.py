# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:41:13 2017

@author: alban_000
"""

import numpy as np
import matplotlib.pyplot as plt

from myData_manager import DataManager
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.confusion_matrix(cm, classes = class_names, normalize = True, title = 'Normalized confusion matrix')

plt.show()
