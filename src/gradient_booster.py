# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:52:01 2019

@author: Parsian Asgari
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class Regressor_GradientBoost:
    '''
    The Regressor_GradientBoost is a Decision Tree based implementation of 
    Gradient Boost. It uses Scikit learn's DecisionTreeRegressor.
    
    Attributes
    ----------
    
    target : Pandas series
        Target's series
    
    features_df : Pandas dataframe
        Features' dataframe
    
    max_depth : int
        The DecisionTreeRegressor's max_depth. This is the max depth 
        of the tree.
    
    ntrees : int
        The maximum number of regressing trees used for gradient boosting.
    
    learning_rate : float
        The weight used for each regressing tree. This value adjusts the 
        strength of the weak learning trees.
    
    Methods
    -------
    
    fit_decisiontree()
        Generates a regressor base on Scikit learn's DecisionTreeRegressor
    
    boost_gradient()
        The gradient boosting aglorithm which utilizes DecisionTreeRegressor 
        as its base predicting model.
    
    predict()
        Predicts target based using boost_gradient() model for a given new set 
        of X (test features)

    squared_error_loss()
        Calculates Squared Error Loss.    
    '''    
    def __init__(self, features_df, target, max_depth=1, ntrees=10, 
                 learning_rate=0.1, min_samples_split=2, min_samples_leaf=1):
        '''
        The constructor for Regressor_GradientBoost
        
        Parameters
        
        ----------
        
        target : Pandas series
            Target's series
    
        features_df : Pandas dataframe
            Features' dataframe
    
        max_depth : int
            The DecisionTreeRegressor's max_depth. This is the max depth 
        of the tree.
    
        ntrees : int
            The maximum number of regressing trees used for gradient boosting.
    
        learning_rate : float
            The weight used for each regressing tree. This value adjusts the 
            strength of the weak learning trees.
        
        min_samples_split : int
            The minimum number of samples required to split an internal node 
            (from DecisionTreeRegressor)
        
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node. 
            (from DecisionTreeRegressor)
        
        
        '''        
        self.X = features_df
        self.y = target
        self.max_depth = max_depth
        self.ntrees = ntrees
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    
    def fit_decisiontree(self, X, y):
        '''
        Generates a regressor base on Scikit learn's DecisionTreeRegressor
        
        Parameters
        ----------
        X : A Pandas DataFrame for features
        
        y : A Pandas DataFrame for target
            
        
        Returns:
            A regressor model base on Scikit-learn's DecisionTreeRegressor
        
        '''            
        try:
            model = DecisionTreeRegressor(max_depth=self.max_depth, 
                                          min_samples_split=self.min_samples_split, 
                                          min_samples_leaf=self.min_samples_leaf)
            model.fit(X, y)
            
        except Exception as e:
            print(e)
            
        return model
    
    
    def boost_gradient(self, X, y):
        '''
        The gradient boosting aglorithm which utilizes DecisionTreeRegressor 
        as its base predicting model.
        
        Parameters
        ----------
        X : A Pandas DataFrame for features
        
        y : A Pandas DataFrame for target
            
        
        Returns:
            first prediction, and a list of models generated via gradient
            boosting.
        
        '''           
        try:
            models = []
            training_rmse = []
            y_pred = np.array([y.mean()]*len(y))
            f0 = y_pred
            rmse = self.rmse(y, f0)
            print('RMSE at first prediction: ', rmse)
        
            for tree in range(self.ntrees):
                residuals = y - y_pred
                fitted = self.fit_decisiontree(X, residuals)
                models.append(fitted)
                fit_pred = fitted.predict(X)
                y_pred += self.learning_rate * fit_pred
                rmse = self.rmse(y, y_pred)
                training_rmse.append(rmse)

                print('RMSE for tree #{} is:'.format(str(tree)), rmse)
                
        except Exception as e:
            print(e)
        
        return f0, models, training_rmse
    
    
    def predict(self, X, f0, models):
        '''
        Predicts target based using boost_gradient() model for a given new set 
        of X (test features)
        
        Parameters
        ----------
        X : A Pandas DataFrame for features (e.g. the X test)
            
        
        Returns:
            predictions for given X (e.g. X Test.
        
        '''        
        try:
            y_pred = np.array([f0[0]]*len(X))
            for model in models:
                y_pred += self.learning_rate * model.predict(X)
            
        except Exception as e:
            print(e)
        
        return y_pred
        
    
    @staticmethod
    def rmse(y, y_pred):
        '''
        Calculates Root Mean Squared Error.
        
        Parameters
        ----------
        y : A Pandas DataFrame for target (e.g. the Y test)

        y_pred : A Pandas DataFrame for predicted target
        
        Returns:
            Returns calculated rmse.
        
        '''        
        mse = ((y-y_pred)**2).mean()
        rmse = np.sqrt(mse)
        
        return rmse


