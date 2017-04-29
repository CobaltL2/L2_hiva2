# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:28:09 2017
@author: nicolas.dauprat
"""
from myData_manager import DataManager
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
        
#test qui ne s'effectue que lorsque l'on execute la classe dans spyder par exemlpe
print "\nTesting DataManager on Iris dataset\n"
#On va tester notre classe en utilisant les données d'Iris
print "Loading the dataset"
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
print "Loading successful"

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print "Predicting using SVM"
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
print "Prediction successful"
#On initialise une instance de la classe
print "Displaying the results"
D = DataManager(y_pred, y_test)
#On test l'affichage
print "\nText results :"
D.afficheTableau()
#Teste creation et affichage de la matrice de confusion
print "\nConfusion matrix :\n"
D.plot_confusion_matrix()