from sklearn.ensemble import RandomForestClassifier
from myPreprocessing import Preprocessor


print "\nTesting preprocessing on our training data:\n"
print "Importing data"
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
print "Data imported successfully \n"

print(data.shape)
Y_train = data['target'].values   
X_train = data.drop('target', axis=1).values
                   
print("*** Original data ***")
print data

Prepro = Preprocessor(RandomForestClassifier(n_estimators=200, max_features="auto"))  
data    = Prepro.fit_transform(X_train,Y_train)    
  
print("*** Transformed data ***")
print data
print("%d * %d" %(len(data) ,len(data[0])))