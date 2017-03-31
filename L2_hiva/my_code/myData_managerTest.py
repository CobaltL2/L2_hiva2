# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:20:42 2017

@author: Nicolas
"""
from DataManager import datamanager
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

#On va tester notre classe en utilisant les données d'Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
#On initialise une instance de la classe
D = datamanager(y_pred, y_test)
#Teste creation et affichage de la matrice de confusion
D.plot_confusion_matrix()
#On test l'affichage
D.affiche()