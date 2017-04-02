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

    def __init__(self,classifier, transformer=identity,debug=False):              
        self.debug=debug             
        self.classifier = classifier
        self.index = []
    
        
    def fit(self, X, y):
        self.scaler=StandardScaler()
        self.lowVar = VarianceThreshold(threshold=(0.05))
        return self

    
    def fit_transform(self, X, y): 
        return self.fit(X,y).transform(X)
    
    def transform(self, X, y=None):         
        if self.debug:
            print X
        if self.index == []:
            X = self.lowVar.fit_transform(X)
            self.index = self.lowVar.get_support()            
            self.length = len(X[0])
            if self.debug:
                print self.length
            return X
        else:
            new_tab = np.empty((len(X),self.length),dtype=float)
            counter = 0
            for i in range (0,len(self.index)):
                if self.index[i] == True:
                    for j in range(0,len(X)):
                        new_tab[j][counter] = X[j][i]
                    counter = counter+1
            if self.debug:
                print new_tab
            return new_tab

        
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