# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:28:09 2017

@author: nicolas.dauprat
"""

#Mettre le chemin vers le code
codedir = '../sample_code'                        
from sys import path
from os.path import abspath
path.append(abspath(codedir))

#Mettre le chemin vers les données
datadir = '../public_data'
dataname = 'hiva'
basename = datadir  + dataname

#importe la classe data-manager présente dans le startingkit afin de créer un héritage
import data_manager
#Importe une classe possédant des méthodes nous permettant d'importer les
#données sous le format Panda Data Frame
import data_io
#importe les outils nous permettant ensuite de créer des figures
import matplotlib.pylab as plt
import numpy as np
import itertools
#Importe ce dont nous avons besoin pour le test et la matrice de confusion
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Notre classe datamanager hérite de celle du starting kit
class DataManager(data_manager.DataManager):
    #Constructeur : pour initier la classe ecrire DataManager(resultatClassifieur, label)
    #le self est présent pour la création d'attributs
    def __init__(self, predictedLabel, label):
        ''' New contructor.'''
        DataManager.__init__(self, basename, datadir)
        self.predictedLabel = predictedLabel
        self.label = label
        
    #charge les données    
    def loadData(self):    
        reload(data_io)
        #Importe les données sous le format Panda Data Frame
        self.data = data_io.read_as_df(datadir)
    
    #affiche les données
    def afficheData(self):    
        self.data.head()
        #affiche la description des données
        self.data.describe()
    
    #affiche la variance des données
    def afficheVariance(self):
        X_train = self.data.drop('target', axis=1).values     
        var_features = np.var(X_train, axis=0)
        #affiche la variance des données sous forme d'une liste
        print var_features
        #affiche cette meme variance sous forme d'un plot de points où chaque point 
        #correspond à une feature
        plt.plot(var_features, 'b*')
        plt.ylabel("Variance")
        plt.xlabel("feature")
        plt.title("Variance des features")
        plt.show()
    
    #Affiche tableau nbErreurs...    
    def afficheTableau(self): 
        #initilaisation
        label = self.label
        pL = self.predictedLabel
        nbErreurs, posAsPos, posAsNeg, negAsPos, negAsNeg, labelPos, labelNeg = 0, 0, 0, 0, 0, 0, 0
        #calcule des variables
        for k in range (len(label)) :
            if (pL[k] != label[k]) :
                nbErreurs += 1
                if (label[k] == 1) :
                    posAsNeg += 1
                    labelPos += 1
                else :
                    negAsPos += 1
                    labelNeg += 1    
            else :
                if (label[k] == 1) :
                    posAsPos += 1
                    labelPos += 1
                else :
                    negAsNeg += 1
                    labelNeg += 1
         
        nbErreurs = (nbErreurs * 100) / len(label)
        posAsPos = posAsPos * 100 / len(label) 
        posAsNeg = posAsNeg * 100 / len(label) 
        negAsNeg = negAsNeg * 100 / len(label) 
        negAsPos = negAsPos * 100 / len(label)
        
        #Création du tableau affichant nbErreurs...
        li = np.array([["NbErrors",nbErreurs],["Positive labeled as positive",posAsPos],
                       ["Positive labeled as negative",posAsNeg],["Negative labeled as negative",negAsNeg],
                       ["Negative labeled as positive",negAsPos]])
        print "\n Raw results (in % of the whole dataset):"
        print li
        
    #Creation matrice de confusion
    def plot_confusion_matrix(self) :
        #creation matrice de confusion
        cm = confusion_matrix(self.label, self.predictedlabel)
        classes = self.data.name
        title = 'Confusion matrix'
        cmap = plt.cm.Blues
        #creation de la heatmap
        plt.imshow(cm, interpolation='nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #Normalisation de la matrice de confusion + modification de la heatmap
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cm)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
        #Affichage de la heatmap
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
#test qui ne s'effectue que lorsque l'on execute la classe dans spyder par exemlpe
if __name__=="__main__":
    #On va tester notre classe en utilisant les données d'Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    #On initialise une instance de la classe
    D = DataManager(y_pred, y_test)
    #Teste creation et affichage de la matrice de confusion
    D.plot_confusion_matrix()
    #On test l'affichage
    D.affiche()

