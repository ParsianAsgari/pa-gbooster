# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:25:12 2019

@author: Parsian Asgari
"""

from src import CSV_Stitcher
from src import FilePaths, Data_Properties
from sklearn.model_selection import train_test_split



class Make_DataSet(CSV_Stitcher):
    '''
    The Make_DataSet, inherited from CSV_Stitcher, is used to stitches all .csv
    files, impute, seperate Target from Features tables into Pandas dataframes. 
    It can also write the processed tables into a stiched .csv file. 
    
    Attributes
    ----------
    
    null_handle_method : str
        '0', 'median', 'mean'
        Methods to fill the missing values
    
    input_path : str
        The input directory where .csv files are present. Path is imported 
        from FilePaths module
    
    output_path : str
        The output directory where the stitched .csv files will be saved. 
        Path is imported from FilePaths module
    
    Methods
    -------
    
    impute_null()
        Imputes the null values for a given method on a Pandas dataframe
    
    extract_features_target()
        Generates two Pandas Series and Dataframe for Target and Features.
        Features, Target properties are imported from data_properties module.
       
    extract_headers()
        Extract .csv table headers

    load_split_data()
        Intakes a Pandas DataFrame as a data source and splits the data table 
        into training set and test set.
    
    df_to_csv
        Writes the X_train, X_test, y_train, y_test tables into a .csv file.
    '''  
    
    def __init__(self, null_handle_method=Data_Properties.null_handle_method, 
                 input_path = FilePaths.path_raw, output_path = FilePaths.path_processed, 
                 test_size=Data_Properties.test_size):
        '''
        The constructor for Make_DataSet
        
        Parameters
        
        ----------
        
        null_handle_method : str
            '0', 'median', 'mean'
            Methods to fill the missing values
        
        input_path : str
            The input directory where .csv files are present.
            Path is imported from data_properties module

        output_path : str
            The output directory where the stitched .csv files will be saved.
            Path is imported from data_properties module
        
        test_size : float
            The size of the test size from 0 to 1 [Default = 0.2] set under appconfig's Data_Properties.set_size
        '''
        super().__init__(input_path, output_path)
        self.null_handle_method = null_handle_method   
        self.test_size = test_size
    
    
    def impute_null(self):
        '''
        impute_null()
            Imputes the null values for a given method
        
        Returns:
            A sorted list of strings.
        '''
        
        try:
            dataframe = self.stitch_csv()
            dataframe = dataframe.drop(['Unnamed: 0'], axis=1)
            
            if str(self.null_handle_method) == '0':
                dataframe = dataframe.fillna(0)
                print('Fill null values with :', self.null_handle_method)
            elif str(self.null_handle_method).lower() == 'mean':
                dataframe = dataframe.apply(lambda x: x.fillna(x.mean()), 
                                            axis=0)
                print('Fill null values with :', self.null_handle_method)
            elif str(self.null_handle_method).lower() == 'median':
                dataframe = dataframe.apply(lambda x: x.fillna(x.median()), 
                                            axis=0)
                print('Fill null values with :', self.null_handle_method)
            elif str(self.null_handle_method).lower() == 'none':
                dataframe = dataframe
                print('Fill null values with :', self.null_handle_method)
                
            else:
                dataframe = dataframe
                print('Imputing method was not detected, only allowed methods are:')
                print('\n')
                print('0', 'median', 'mean', 'none')
                print('\n')
                print('Data is unchanged')
                
        except Exception as e:
            print(e)
        
        return dataframe
    
    
    def extract_features_target(self):
        '''
        extract_features_target()
            Generates two Pandas dataframe for Target and Features. 
            Features, Target properties are imported from data_properties module.
        
        Returns:
            Two Features dataframe, Target series 
        '''        
        try:
            dataframe = self.impute_null()
            features_df = dataframe[Data_Properties.features]
            target_sr = dataframe[Data_Properties.target]
            
            print('The target is: ', Data_Properties.target)
            print('The features are: ', Data_Properties.features)
            
        except Exception as e:
            print(e)
        
        return features_df, target_sr


    def load_split_data(self):
        '''
        Intakes a Pandas DataFrame as a data source and splits the data table 
        into training set and test set.
        
        Parameters
        ----------
        dataframe : A Pandas DataFrame 
            Intakes the data as a Pandas DataFrame 
        
        Returns:
            Splited dataframes for training and testing (X_train, X_test, 
            y_train, y_test)
        
        '''    
        try:
            X, y = self.extract_features_target()
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=self.test_size)
            
            print('Generated X_train, X_test, y_train, y_test with test_size: ', self.test_size)
        except Exception as e:
            print(e, 'Only Pandas dataframe is accepted as an input')
            
        return X_train, X_test, y_train, y_test
    
    
    def df_to_csv(self):
        '''
        df_to_csv()
            Writes the processed Pandas dataframe into four .csv files:
                - X_train.csv
                - y_train.csv
                - X_test.csv
                - y_test.csv
        
        Returns:
            Four .csv files, X_train.csv, y_train.csv, X_test.csv, y_test.csv
        '''              
        try:
            print('\n')
            print('Spliting data into X_train, X_test, y_train, y_test with test size:', self.test_size)
            print('\n')
            X_train, X_test, y_train, y_test = self.load_split_data()
            
            print('Writing X_train table to .csv')
            self.write_stitched_csv(X_train, output_fname='X_train.csv')
            print('Writing y_train table to .csv')
            self.write_stitched_csv(y_train, output_fname='y_train.csv')
            print('Writing X_test table to .csv')
            self.write_stitched_csv(X_test, output_fname='X_test.csv')
            print('Writing y_test table to .csv')
            self.write_stitched_csv(y_test, output_fname='y_test.csv')
            
            print('Writing is done, the .csv files can be found here: ', self.output_path)
            
        except Exception as e:
            print(e)



            
            
    
    
    