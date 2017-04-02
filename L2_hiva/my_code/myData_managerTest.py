# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:20:42 2017

@author: Nicolas
"""
#ATTENTION, avant d'executer ce code, veuillez vérfifier que la VERSION DE SKLEARN utilisée est bien 0.18.1 ou plus.

from myData_manager import DataManager
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

#On va tester notre classe en utilisant les données d'Iris
class DataManagerTest():
  #constructeur qui initialise une instance de myData_manager
  def __init__(self):
  '''Constructeur'''
    #initialisation les données d'Iris
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    class_names = self.iris.target_names
    #On applique un classifieur sur les données d'Iris
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    classifier = svm.SVC(kernel='linear', C=0.01)
    Y_pred = classifier.fit(X_train, Y_train).predict(X_test)
    #On initialise une instance de la classe
    self.D = DataManager(Y_pred, Y_test)
    
  #Test de la matrice de confusion avec les données Iris
  def cmapTest(self): 
    #Teste creation et affichage de la matrice de confusion
    self.D.plot_confusion_matrix()
  
  #On test la méthode afficheTableau
  def afficheTableauTest(self):
    #On test la calcule des variables et l'affichage du tableau
    self.D.afficheTableau()
    
  #On test la methode loadData
  def loadDataTest(self):
    iris = datasets.load_iris()
    iris.data.head()
    iris.data.describe()

#test
if __name__=="__main__":
  #initialisation
  test=DataManagerTest()
  #test chargement des données
  test.loadDataTest()
  #test affichage tableau de valeurs
  test.afficheTableauTest()
  #test creation et affichage de la matrice de confusion sous forme de heatmap
  test.cmapTest()
    
