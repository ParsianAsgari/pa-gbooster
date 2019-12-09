# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:07:29 2019

@author: Parsian Asgari
"""
import numpy as np
import os.path
import pandas as pd
import pickle
import xgboost as xgb
     
from sklearn import metrics


from pa_app_config import FilePaths, RegressGB_Parameters, RegressXGB_Parameters, Data_Properties
from pa_ml_utils import Regressor_GradientBoost

class Model_Trainer:
    '''
    This class allows to select training model, load data, train and save trained model.
    
    Attributes:
    -----------
    trainer : str
        Select the training method.
        Options: 
            xgb - XGBoost Regressor (default)
            pa-gb - Parsian's Gradient Boost Regressor
    
    save_model_name : str
        File name for the trained model to be saved as a serialized .pkl file.
    
    data_input_path : str
        The path to the X_train.csv, y_train.csv [Default: ./data/processed]  - Can be changed from appconfig.py -> FilePaths.path_processed

    Methods:
    --------
        load_training_data():
            loads X_train.csv, y_train.csv
        
        train_regressor_gradientboost():
            Parsian's DecisionTree based GB Regressor
        
        train_regressor_xgradientboost():
            XGBoost's Regressor
        
        save_model():
            Saves the trained model as a serialized .pkl file under [Default: ./models/]
        
        select_training_model():
            Model selector - intakes model name and trains the model
        
    '''
    
    def __init__(self, target=Data_Properties.target, trainer='xgb', save_model_name='xgb_trained-model',data_input_path = FilePaths.path_processed):
        '''
        Attributes:
        -----------
        target: str
            Target column header
            
        trainer : str
            Select the training method.
            Options: 
                xgb - XGBoost Regressor (default)
                pa-gb - Parsian's Gradient Boost Regressor
    
        save_model_name : str
            File name for the trained model to be saved as a serialized .pkl file.
    
        data_input_path : str
            The path to the X_train.csv, y_train.csv [Default: ./data/processed]  - Can be changed from appconfig.py -> FilePaths.path_processed
        '''
        
        self.target = target
        self.trainer = trainer
        self.data_input_path = data_input_path
        self.save_model_name = save_model_name
        
    
    def load_training_data(self):
        '''
        Loads training data X_train.csv, y_train.csv
        
        returns X_train dataframe, y_train series
        '''
        
        X_train = pd.read_csv(os.path.join(self.data_input_path, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(self.data_input_path, 'y_train.csv'), names=[self.target])
        y_train = y_train[self.target]
        
        return X_train, y_train
    
    
    @staticmethod
    def train_regressor_gradientboost(X_train, y_train):
        '''
        Trains a regressor base on Parsian's pa_ml_utils.Regressor_GradientBoost
            
        Parameters
        ----------
        X_train : A Pandas DataFrame for features
            
        y_train : A Pandas DataFrame for target
                
            
        Returns:
            A regressor model base on pa_ml_utils.Regressor_GradientBoost
            
        '''     
        try:            
            
            print('Initiating the training process using \'pa-gb\' model')
            
            regressor = Regressor_GradientBoost(features_df = X_train, 
                                                target = y_train,
                                                max_depth = RegressGB_Parameters.max_depth, 
                                                ntrees = RegressGB_Parameters.ntrees, 
                                                learning_rate = RegressGB_Parameters.learning_rate)
        
            f0, models, training_rmse = regressor.boost_gradient(X_train, y_train)
            print('Training completed')
    
        except Exception as e:
            print('modeller.py Model_Trainer.train_regressor_gradientboost(): ',e)
            
        
        return f0, models    
    
    
    @staticmethod
    def train_regressor_xgradientboost(X_train, y_train):
        '''
        Generates a regressor base on xgb (XGBoos)
            
        Parameters
        ----------
        X_train : A Pandas DataFrame for features
            
        y_train : A Pandas DataFrame for target
                
            
        Returns:
            A regressor model base on xgb
            
        '''        
        try:
                   
            train_dmatrix = xgb.DMatrix(X_train, label=y_train)
            
            print('Initiating the training process using \'xgb\' model')
            
            fit_model = xgb.train(RegressXGB_Parameters.params, train_dmatrix, num_boost_round=RegressXGB_Parameters.num_boost_round)
            print('Training completed')
        
        except Exception as e:
            print('modeller.py Model_Trainer.train_regressor_xgradientboost(): ',e)
    
        return fit_model
      
    
    def save_model(self, model):
        '''
        Serializes and saves an input model under FilePaths.path_models
            
        Parameters
        ----------
        model : A function's output
            
        modelname : The model's .pkl name (str)
                
            
        Returns:
            None - Saves the model in the FilePaths.path_models directory
            
        ''' 
        print('Serializing the model and saving') 
          
        try:
            
            with open(os.path.join(FilePaths.path_models, self.save_model_name), 'wb') as file:
                pickle.dump(model, file)
                
        except Exception as e:
            print('modeller.py Model_Trainer.save_model(): ',e)
        
        print('Model {modelname} is saved under: {fpath}'.format(modelname=self.save_model_name, fpath=FilePaths.path_models))


    def select_training_model(self):
        
        try:
            
            X_train, y_train = self.load_training_data()
    
            if str(self.trainer).lower() == 'pa-gb':
                fit_model = self.train_regressor_gradientboost(X_train, y_train)
                if str(self.save_model_name) == 'xgb_trained-model':
                    save_model_name = 'pa-gb_trained-model'
                    print('No model name was provided for pa-gb, saving the model as:', save_model_name)
            else:
                fit_model = self.train_regressor_xgradientboost(X_train, y_train)
                
            self.save_model(fit_model)
    
        except Exception as e:
            print('modeller.py Model_Trainer.select_training_model(): ',e)


class Model_Predictor:
    '''
    This class allows to select predictor model, load trained model, test data.
    
    Attributes:
    -----------

    predictor : str
        Select the predictor method.
        Options: 
            xgb - XGBoost Regressor (default)
            pa-gb - Parsian's Gradient Boost Regressor
    
    load_model_name : str
        File name for the trained model to be loaded as a serialized .pkl file.
    
    data_input_path : str
        The path to the X_test.csv, y_test.csv [Default: ./data/processed]  - Can be changed from appconfig.py -> FilePaths.path_processed

    Methods:
    --------
        load_testing_data():
            loads X_test.csv, y_test.csv
        
        predict_regressor_gradientboost():
            Predict using Parsian's DecisionTree based GB Regressor
        
        predict_regressor_xgradientboost():
            Predict using XGBoost's Regressor
        
        load_model():
            loads the trained model as a serialized .pkl file under [Default: ./models/]
        
        select_predicting_model():
            Model selector - intakes model name and predicts using the model
        
    '''
    
    def __init__(self, target=Data_Properties.target, predictor='xgb', load_model_name='xgb_trained-model',data_input_path = FilePaths.path_processed):
        '''
        Attributes:
        -----------
        target: str
            Target column header
            
        predictor : str
            Select the predicting method.
            Options: 
                xgb - XGBoost Regressor (default)
                pa-gb - Parsian's Gradient Boost Regressor
    
        load_model_name : str
            File name for the trained model to be loaded as a serialized .pkl file.
    
        data_input_path : str
            The path to the X_train.csv, y_train.csv [Default: ./data/processed]  - Can be changed from appconfig.py -> FilePaths.path_processed
        '''
        self.target = target
        self.predictor = predictor
        self.data_input_path = data_input_path
        self.load_model_name = load_model_name
        
    
    def load_testing_data(self):
        '''
        Loads training data X_train.csv, y_train.csv
        
        returns X_train dataframe, y_train series
        '''
        
        X_test = pd.read_csv(os.path.join(self.data_input_path, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(self.data_input_path, 'y_test.csv'), names=[self.target])
        y_test = y_test[self.target]
        return X_test, y_test
    
    
    @staticmethod
    def predict_regressor_gradientboost(X_test, y_test, f0, reg_gb_models):
        '''
        Generates a prediction base on Parsian's 
        pa_ml_utils.Regressor_GradientBoost model
            
        Parameters
        ----------
        X_test : A Pandas DataFrame for features
            
        y_test : A Pandas Series for target
                
            
        Returns:
            Predictions base on reg_gb_models generated from pa_ml_utils.Regressor_GradientBoost
            
        '''        
        try:
            from pa_ml_utils import Regressor_GradientBoost
            
            print('Initiating the prediction process using \'pa-gb\' model')
            
            regressor = Regressor_GradientBoost(X_test, y_test)
            prediction = regressor.predict(X_test, f0, reg_gb_models)
            rmse = regressor.rmse(y_test, prediction)
            
            print('Prediction completed')
            
        except Exception as e:
            print('modeller.py Model_Predictor.predict_regressor_gradientboost(): ',e)
            
        print("The RMSE for the Regressor Gradient Boost model: ", rmse)
        return prediction, rmse   
    
    
    @staticmethod
    def predict_regressor_xgradientboost(X_test, y_test, reg_xgb_model):
        '''
        Generates a prediction base on a xgb input model (reg_xgb_model)
            
        Parameters
        ----------
        X_test : A Pandas DataFrame for features
            
        y_test : A Pandas DataFrame for target
                
            
        Returns:
            Prediction base on reg_xgb_model generated from xgb
            
        '''      
        try:
            
            print('Initiating the prediction process using \'xgb\' model')
    
            test_dmatrix = xgb.DMatrix(X_test)
            prediction = reg_xgb_model.predict(test_dmatrix)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, prediction))
    
            print('Prediction completed')
        
        except Exception as e:
            print('modeller.py Model_Predictor.predict_regressor_xgradientboost(): ',e)
        
        print("The RMSE for the Regressor XGBoost model: ", rmse)
        
        return prediction, rmse
      
    
    def load_model(self):
        '''
        Loads serialized input model under FilePaths.path_models
            
        Parameters
        ----------
        model : A function's output
        
            
        Returns:
            loaded_model - Model loaded from the FilePaths.path_models directory
            
        ''' 
        print('Loading model')       
        try:
            
            with open(os.path.join(FilePaths.path_models, self.load_model_name), 'rb') as file:
                loaded_model = pickle.load(file)
                
        except Exception as e:
            print('modeller.py Model_Predictor.load_model(): ',e)
        
        print('Model {modelname} is loaded from : {fpath}'.format(modelname=self.load_model_name, fpath=FilePaths.path_models))
        
        return loaded_model


    def select_predicting_model(self):
        
        print('Predictor {} was selected'.format(str(self.predictor)))
        
        try:
            
            X_test, y_test = self.load_testing_data()
            if str(self.predictor).lower() == 'pa-gb':
                f0, models = self.load_model()
                prediction, rmse = self.predict_regressor_gradientboost(X_test, y_test, f0, models)
            else:
                model_loaded = self.load_model()
                prediction, rmse = self.predict_regressor_xgradientboost(X_test, y_test, model_loaded)     
    
        except Exception as e:
            print('modeller.py Model_Predictor.select_predicting_model(): ',e)
    
        return prediction, rmse





        

    
    