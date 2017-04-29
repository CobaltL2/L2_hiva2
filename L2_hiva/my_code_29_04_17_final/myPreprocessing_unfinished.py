import numpy as np
from sklearn import svm
from classifier import Classifier
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
    classifier = Classifier() 
    n_pca=1
    n_skb=1 

    def __init__(self, transformer=identity):              
        self.transformer = self    
        
    def fit(self, X, y):
        "Selection des hyperparametre"
        PCAPip = Pipeline([('pca',PCA()), ('clf',self.classifier)])    
        t = []
        i=10
        while i <int(200):
            t.append(i)         
            i+=2
        grid_search = GridSearchCV(PCAPip,{'pca__n_components' : t},verbose=3,scoring=make_scorer(accuracy_score))
        grid_search.fit(X,y)
        self.n_pca=grid_search.best_params_.get('pca__n_components')
        return self 
    
    def fit_transform(self, X, y): 
        return self.fit(X,y).transform(X)
    
    def transform(self, X, y=None):         
        "Suppression des features aux variances les plus faibles"
        sel = VarianceThreshold(threshold=(0.05))
        X = sel.fit_transform(X)      
        "Selection des donnees avec les algorithmes PCA"
        pca = PCA(n_components=self.n_pca)
        X=pca.fit_transform(X)   
        "Standardization des donnees"
        scaler=StandardScaler()
        X=scaler.fit_transform(X)  
        return X  
        
if __name__=="__main__":
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

    print(data.shape)
    Y_train = data['target'].values   
    X_train = data.drop('target', axis=1).values
                       
    print("*** Original data ***")
    print data

    Prepro = Preprocessor()  
    data    = Prepro.fit_transform(X_train,Y_train)    
  
    print("*** Transformed data ***")
    print data
    print("%d * %d" %(len(data) ,len(data[0])))