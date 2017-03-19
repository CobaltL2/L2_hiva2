codedir = 'my_code/'  
from sys import path; path.append(codedir)
datadir = '../public_data/'                        # Change this to the directory where you put the input data
dataname = 'hiva'
basename = datadir  + dataname
# !ls $basename*
import data_io
reload(data_io)
data = data_io.read_as_df(basename)                           # The data are loaded as a Pandas Data Frame
#data.to_csv(basename + '_train.csv', index=False)           # This allows saving the data in csv format                 
import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np
from sklearn import svm
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
    classifier = svm.SVC() 
    """<Choix  CLASSIFIER>"""
    n_pca=1
    n_skb=1


    def __init__(self, transformer=identity):              
        self.transformer = self    
        
    def fit(self, X, y=None):
        y_train = X['target'].values   
        X_train = X.drop('target', axis=1).values
        
        "Selection des hyperparametre"
        PCAPip = Pipeline([('pca',PCA()),('SKB',SelectKBest()), ('clf',self.classifier)])    
        self.classifier.fit(X_train,y_train)
        t = []
        i=20
        while i <int(100):
            t.append(i)         
            i+=15            
        tab = []      
        for j in range(5,20):
            tab.append(j+1)
        grid_search = GridSearchCV(PCAPip,{'pca__n_components' : t,'SKB__k' : tab},verbose=1,scoring=make_scorer(accuracy_score))
        grid_search.fit(X_train,y_train)
        self.n_pca=grid_search.best_params_.get('pca__n_components')
        self.n_skb=grid_search.best_params_.get('SKB__k')
        return self 
    
    def fit_transform(self, X, y=None): 
        return self.fit(X).transform(X)
    
    def transform(self, X, y=None):        
        y_train = X['target'].values   
        X_train = X.drop('target', axis=1).values
        
        "Suppression des features aux variances les plus faibles"
        sel = VarianceThreshold(threshold=(0.05))
        X_train = sel.fit_transform(X_train)  
        
        "Standardization des données"
        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)       
        
        "Selection des données avec les algorithmes PCA et SelectKBest"
        pca = PCA(n_components=self.n_pca)
        kbest=SelectKBest(k=self.n_skb)
        
        X_train=pca.fit_transform(X_train)  
        X_train=kbest.fit_transform(X_train,y_train) 
        "Transformation des données terminée"       
        return X_train   
        
if __name__=="__main__":
    print(data.shape)
    Y_train = data['target'].values   
    X_train = data.drop('target', axis=1).values
    print("*** Original data ***")
    print data

    Prepro = Preprocessor()  
    data    = Prepro.fit_transform(data)    
  
    print("*** Transformed data ***")
    print data
    print(len(data),"x",len(data[0]))