#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:31:38 2017

@author: sabrina
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from data_manager import DataManager

class BestParams(GridSearchCV):
    def __init__(self):
        pass
    
    def bestParamsRFC(self):
        D = DataManager("hiva","../public_data")
        X = D.data['X_train']
        y = D.data['Y_train']
        param_grid = {
                'n_estimators': [10, 20, 50, 100, 200], 
                'max_features': ['auto', 'sqrt', 'log2']}
        
        clf = GridSearchCV(RandomForestClassifier(), param_grid = param_grid)
        clf.fit(X, y)
        
        # Sélection des paramètres les plus performants
        best_n_estimators = clf.best_params_['n_estimators']
        best_max_features = clf.best_params_['max_features']
        
        print "Best n_estimators = {}".format(best_n_estimators)
        print "Best max_features = {}".format(best_max_features)
        
        return [best_n_estimators, best_max_features]
