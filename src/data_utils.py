# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:33:46 2019

@author: Parsian Asgari
"""
import glob
import missingno as msno
import os
import pandas as pd
import re

from IPython.display import display


class CSV_Stitcher:
    '''
    The CSV_Stitcher is used to stitches all .csv files in a given input folder
    and outputs the stitched .csv file to the provided output folder.
    
    Attributes
    ----------
    
    input_path : str
        The input directory where .csv files are present
    
    output_path : str
        The output directory where the stitched .csv files will be saved
    
    Methods
    -------
    
    extract_headers()
        Extract .csv table headers
    
    stitch_csv()
        Stitches the .csv files into Pandas Dataframe
    
    write_stitched_csv
        Writes the stitched tables into a .csv file.
    
    '''
    

    def __init__(self, input_path, output_path):
        
        '''
        The constructor for CSV_Stitcher
        
        Parameters
        
        ----------
        
        input_path : str
            The input directory where .csv files are present

        output_path : str
            The output directory where the stitched .csv files will be saved
        '''
        
        self.input_path = input_path
        self.output_path = output_path
        
    
    @staticmethod
    def sort_list(ls):
        '''
        The function sorts a list of strings depending on the element composition.
        Prioritize sort on numberical values in the string elements.
        
        Parameters
        ----------
        ls : python list
            The name of the output file
        
        Returns:
            A sorted list of strings.
            
        '''
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(ls, key=alphanum_key)        
    
    
    def list_csv_files(self):
        '''
        The function to list all .csv files found in the input_path directory
        
        Returns:
            list of .csv files found
        
        '''
        ext = 'csv'
        os.chdir(self.input_path)
        csv_list = [csvfile for csvfile in glob.glob('*.{}'.format(ext))]
        
        
        return CSV_Stitcher.sort_list(csv_list)
    
    
    def stitch_csv(self):
        '''
        The function reads all .csv files in the input_path directory
        and stitches them into a Pandas dataframe.
        
        Returns:
            A Pandas Dataframe consisting of stitched data.
        
        '''
            
        csv_list = self.list_csv_files()
        if len(csv_list) != 0:
            print('\n')
            print('The following .csv files will get stitched: ')
            print(csv_list)
        
            additional_missing_values = ["n/a", "na", "--", "inf", "-inf", 
                                         "INF", "-INF", "INFINITY", "-INFINITY", 
                                         "- inf", "- INF", " inf", " INF"]
        
            try:
                stitched_csv = pd.concat([pd.read_csv(fname, na_values = additional_missing_values) for fname in csv_list])
            except Exception as e:
                print(e)
        
            print('Stitching is done!')
        
        else:
            print('***' + 'No .csv files were found! Please make sure the data are in the right folder' + '***')
            print('\n')
                
            
        return stitched_csv
    

    def write_stitched_csv(self, dataframe, output_fname='stitched_data.csv', encode='utf-8-sig'):
        '''
        The function writes the stitched dataframe into a .csv file in the output file directory.
        
        Parameters
        ----------
        output_fname : str
            The name of the output file
        
        encode : str
            The type of encoding used by Pandas .to_csv() method
        
        Returns:
            A stitched data .csv file.
        
        '''
        
        dataframe.to_csv(os.path.join(self.output_path, output_fname), index=False, encoding=encode)
        print('Stitched CSV file is ready. It is located at {}'.format(str(self.output_path)))
        


            
class Data_Profiler:
    '''
    The Data_Profiler is used to profile Pandas self.df
    
    Attributes
    ----------
    
    self.df : Pandas self.df
        The input Pandas self.df
    
    Methods
    -------
    
    missing_zero_values_table()
        provide statistics on missing values and zero values
    
    profile_data()
        profiles the dataframe and returns a list of columns with misssing data.
    '''    
    def __init__(self, dataframe):
        self.df = dataframe
    
    
    def missing_zero_values_table(self):
        try:
            zero_val = (self.df == 0.00).astype(int).sum(axis=0)
            mis_val = self.df.isnull().sum()
            mis_val_percent = 100 * self.df.isnull().sum() / len(self.df)
            mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

            mz_table = mz_table.rename(
                    columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
            
            mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
            mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(self.df)
            
            mz_table['Data Type'] = self.df.dtypes
            mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
            print ("Your selected dataframe has " + str(self.df.shape[1]) + 
                   " columns and " + str(self.df.shape[0]) + " Rows.\n"      
                   "There are " + str(mz_table.shape[0]) +
                   " columns that have missing values.")
        except Exception as e:
            print(e)
        display(mz_table)

    def profile_data(self, tablename=None):
        '''
        The function profiles the dataframe and returns a list of columns with missing values
        
        Parameters
        ----------
        tablename : str
            The name of the dataframe
        
        Returns:
            Profiles the data and returns a list of columns with missing values
        
        '''
        try:        
            missing_value_columns = self.df.columns[self.df.isnull().any()].tolist()
            plot = msno.matrix(self.df[missing_value_columns])
            print('\033[1m' + '\033[4m' + 'Beginning of summary for {}: '.format(str(tablename)) + '\033[0m')
            print('\n')    
            print('\033[1m' + 'Data columns of {}: '.format(str(tablename)) + '\033[0m')
            print('\n')
            display(self.df.info())#self.df.columns
            print('\n')
            print(self.missing_zero_values_table())
            print('\n')
            print('\033[1m' + 'Missing data in following columns of {}: '.format(str(tablename)) + '\033[0m')
            print(self.df.columns[self.df.isnull().any()].tolist())
            print('\n')
            print('\n')
            print('\033[1m' + 'Summary statistics of {}: '.format(str(tablename)) + '\033[0m')
            print('\n')
            display(self.df.describe(include='all'))
            print('\n')
            print('\033[1m' + '\033[4m' + 'First 5 rows of {}: '.format(str(tablename)) + '\033[0m')
            display(self.df.head(5))
            print('\n')
            print('\n')
            print('\033[1m' + "Frequency of missing data across the table indices" + '\033[0m')
            print('\n')
            display(plot)
        except Exception as e:
            print(e)
        return missing_value_columns