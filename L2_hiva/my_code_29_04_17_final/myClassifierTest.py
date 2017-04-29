# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:18:11 2017

@author: albanpetit
"""

from myData_manager import DataManager
from classifier import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score

input_dir = "../public_data"
output_dir = "../res"
basename = 'HIVA'
    #Initialisation de DataManager et du Classifier
print "Initializing DataManager and Classifier"
D = DataManager([],[])
C = Classifier()

print "Testing classifier"

Y_train_data = D.data['Y_train']
print "-----Fitting-----"
print "Fitting on training data"
C.fit(D.data['X_train'], Y_train_data)
print "Fitting successful\n"
print "-----Predicting-----"
print "Predicting on training data"
Y_train_predict = C.predict(D.data['X_train'])
print "Prediction successful\n"
print "Predicting probabilities on training data"
Y_train_predictproba = C.predict_proba(D.data['X_train'])
print "Prediction successful\n"
print "Predicting on validation data"
Ypredict_valid = C.predict(D.data['X_valid'])
print "Prediction successful\n"
print "Predicting on testing data"
Ypredict_test = C.predict(D.data['X_test'])
print "Prediction successful\n"
print "-----Scores-----"
print "Calculating accuracy"
accuracy_train = accuracy_score(Y_train_data, Y_train_predict)
print "Accuracy : %5.2f\n" % (accuracy_train)
print "Calculating cross-val score"
accuracy_crossval = cross_val_score(C, D.data['X_train'], Y_train_data, cv=5, scoring='accuracy')
print "Cross-validation score = %5.2f +-%5.2f \n" % (accuracy_crossval.mean(), accuracy_crossval.std())
print "Confusion matrix"
print confusion_matrix(Y_train_data, Y_train_predict)