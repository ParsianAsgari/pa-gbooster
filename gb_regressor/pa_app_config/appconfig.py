# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:43:14 2019

@author: Parsian Asgari
"""

import os.path


class FilePaths:
    '''
    File path attributes are accessible from this class.
    '''

    parentdir = os.path.abspath('..')
    path_raw = os.path.join(parentdir,'data','raw')
    path_processed = os.path.join(parentdir,'data','processed') 
    path_models = os.path.join(parentdir, 'models')
    

class RegressGB_Parameters:
    
    '''
    Model parameters for Regressor_GradientBoost:
    '''
    
    max_depth = 2
    ntrees = 250
    learning_rate = 0.1
    min_samples_split = 2
    min_samples_leaf = 1
    


class RegressXGB_Parameters:
    
    '''
    Model parameters for xgb:
    '''
    
    params = {'eval_metric': 'rmse',
              'max_depth': 7,
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': 0,
              'colsample_bytree': 0.9}
    
    num_boost_round=200
    
    
class Data_Properties:
    
    '''
    Data columns for target and features.
    
    Attributes:
    -----------
    
    null_handle_method : str
        This value sets the method to fill the null values. Options: 'median', 'mean', '0'
    
    test_size : float
        this is the fraction of the data to be used for generating test data. [default: 0.2 (20%)]
    
    target : str
        This is the column header for the target value
    
    features : list
        This is the list of feature column headers.
        
    '''    
    null_handle_method = 'median'
    
    test_size = 0.2

    target =   'target'
    
    features = ['feature_1', 
                'feature_2', 
                'feature_3', 
                'feature_4', 
                'feature_5', 
                'feature_6', 
                'feature_7', 
                'feature_8', 
                'feature_9', 
                'feature_10', 
                'feature_11', 
                'feature_12', 
                'feature_13', 
                'feature_14', 
                'feature_15', 
                'feature_16', 
                'feature_17', 
                'feature_18', 
                'feature_19', 
                'feature_20', 
                'feature_21', 
                'feature_22', 
                'feature_23', 
                'feature_24', 
                'feature_25', 
                'feature_26', 
                'feature_27', 
                'feature_28', 
                'feature_29', 
                'feature_30', 
                'feature_31', 
                'feature_32', 
                'feature_33', 
                'feature_34', 
                'feature_35', 
                'feature_36', 
                'feature_37', 
                'feature_38', 
                'feature_39', 
                'feature_40', 
                'feature_41', 
                'feature_42', 
                'feature_43', 
                'feature_44', 
                'feature_45', 
                'feature_46', 
                'feature_47', 
                'feature_48', 
                'feature_49', 
                'feature_50', 
                'feature_51', 
                'feature_52', 
                'feature_53', 
                'feature_54', 
                'feature_55', 
                'feature_56', 
                'feature_57', 
                'feature_58', 
                'feature_59', 
                'feature_60', 
                'feature_61', 
                'feature_62', 
                'feature_63', 
                'feature_64', 
                'feature_65']
