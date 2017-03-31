# -*- coding: utf-8 -*-
"""
@author: kevin.ah-son
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.feature_selection import VarianceThreshold,SelectKBest

def identity(x): 
    return x 

class Preprocessor(BaseEstimator):    
    """<Choix  CLASSIFIER>"""
    n_pca=1
    n_skb=1 

    def __init__(self, classifier, transformer=identity):              
        self.classifier = classifier
        self.transformer = self
        self.index = []
        
    """def fit(self, X, y):
        self.y_train= y    
        "Selection des hyperparametre"
        PCAPip = Pipeline([('pca',PCA()),('SKB',SelectKBest()), ('clf',self.classifier)])    
        self.classifier.fit(X,self.y_train)
        t = []
        i=20
        while i <int(60):
            t.append(i)         
            i+=15            
        tab = []      
        for j in range(5,20):
            tab.append(j+1)
        grid_search = GridSearchCV(PCAPip,{'pca__n_components' : t,'SKB__k' : tab},verbose=1,scoring=make_scorer(accuracy_score))
        grid_search.fit(X,self.y_train)
        self.n_pca=grid_search.best_params_.get('pca__n_components')
        self.n_skb=grid_search.best_params_.get('SKB__k')
        return self""" 
    
    def fit_transform(self, X, y): 
        return self.transform(X)
        """return self.fit(X,y).transform(X)"""
    
    def transform(self, X, y=None):   
        "Suppression des features aux variances les plus faibles"
        if self.index == []:          
            sel = VarianceThreshold(threshold=(0.05))
            X = sel.fit_transform(X)
            self.index = sel.get_support()
            self.length = len(X[0])
            return X
        else:
            new_tab = np.empty((len(X),self.length),dtype=float)
            counter = 0
            for i in range (0,len(self.index)):
                if self.index[i] == True:
                    for j in range(0,len(X)):
                        new_tab[j][counter] = X[j][i]
                    counter = counter+1
            return new_tab
                    
        """ "Selection des donnees avec les algorithmes PCA et SelectKBest"
        pca = PCA(n_components=self.n_pca)
        kbest=SelectKBest(k=self.n_skb)
        
        X=pca.fit_transform(X)  
        X=kbest.fit_transform(X,self.y_train)

        "Standardization des donnees"
        scaler=StandardScaler()
        X=scaler.fit_transform(X)"""
     
        
        #Exemple de preprocessing quand on exécute la classe seule
if __name__=="__main__":
    codedir = '../sample_code/'  
    from sys import path; path.append(codedir)
    datadir = '../public_data/'                        # Change this to the directory where you put the input data
    dataname = 'hiva'
    basename = datadir  + dataname
    # !ls $basename*
    import data_io
    reload(data_io)
    data = data_io.read_as_df(basename)                           # The data are loaded as a Pandas Data Frame
    #data.to_csv(basename + '_train.csv', index=False)           # This allows saving the data in csv format                 

    print(data.shape)
    Y_train = data['target'].values   
    X_train = data.drop('target', axis=1).values
                       
    print("*** Original data ***")
    print data

    Prepro = Preprocessor(RandomForestClassifier(n_estimators=150))  
    data    = Prepro.fit_transform(X_train,Y_train)    
  
    print("*** Transformed data ***")
    print data
    print("%d * %d" %(len(data) ,len(data[0])))
