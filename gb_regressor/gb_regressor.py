"""GB_Regressor - Parsian Asgari's Gradient Booster Regressor

Usage:
    gb_regressor.py splitdata <test-size>
    gb_regressor.py train <trainer-type> <save-model-name> 
    gb_regressor.py predict <predictor-type> <load-model-name>
    gb_regressor.py (-h | --help)

Arguments:
    <test-size>        Choose the fraction size of the test data. Between: 0,10 [default: 0.2]
    <trainer-type>     Choose Gradient Booster Trainer: pa-gb, xgb. [default: xgb]
    <predictor-type>   Choose Gradient Booster Predictor: pa-gb, xgb. [default: xgb]
    <save-model-name>  Serialized model file name to save.
    <load-model-file>  Serialized model file.

Options:
    -h --help          Show this screen.
"""
from docopt import docopt


def dataset(test_size):
    '''
    Generates X_train, X_test, y_train, y_test data from raw datasets
    located under ./data/raw (set under: appconfig.py -> FilePaths.path_raw)
    This method automatically stitches data, fills the missing data with median value
    of the column. To change fill behaviour: appconfig.py -> Data_Properties.null_handle_method
    
    Returns:
        X_train.csv, X_test.csv, y_train.csv, y_test.csv

    '''

    try:
        from pa_data_set import Make_DataSet
        
        test_size = float(test_size)
        
        make_dataset = Make_DataSet(test_size=test_size)
        make_dataset.df_to_csv()
        

    except Exception as e:
        print('gb_regressor.py dataset()', e)


def train(trainer, save_model_name):
    '''
    Starts training on data saved under FilePaths.path_models [default: ./data/raw/] and generate a serialized trained model in .pkl format.
    The file is located under ./models (set under: appconfig.py -> FilePaths.path_models)

    Parameters
    ----------

    trainer: Trainer method. Two options: xgb (XGBoost), pa-gb (Parsian's) [default: xgb]

    save_model_name : Name of the serialized file that is going to get saved. [default: xgb_trained-model]
    
    Returns:
        fit_model - trained model

    '''
    print('Trainer {} was selected'.format(str(trainer)))
    
    from pa_train_predict import Model_Trainer
    
    model_trainer = Model_Trainer(trainer=trainer, save_model_name=save_model_name)
    fit_model = model_trainer.select_training_model()

    return fit_model


def predict(predictor, load_model_name):
    '''
    Generates predictions from a serialized trained model in .pkl format.
    The file is located under ./models (set under: appconfig.py -> FilePaths.path_models)

    Parameters
    ----------

    predictor: Predictor method. Two options: xgb (XGBoost), pa-gb (Parsian's) [default: xgb]

    load_model_name : Predictor method. Two options: xgb (XGBoost), pa-gb (Parsian's) [default: xgb_trained-model]
    
    Returns:
        prediction, rmse : Predictions and Root Mean Square Error

    '''
    from pa_train_predict import Model_Predictor
    model_predictor = Model_Predictor(predictor=predictor, load_model_name=load_model_name)
    prediction, rmse = model_predictor.select_predicting_model()

    return prediction, rmse


def main():

    arguments = docopt(__doc__)
    
    import os.path
    import sys
    

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    if arguments['train']:
        train(arguments['<trainer-type>'], 
              arguments['<save-model-name>'])

    elif arguments['predict']:
        predict(arguments['<predictor-type>'],
                arguments['<load-model-name>'])

    elif arguments['splitdata']:
        dataset(arguments['<test-size>'])


if __name__ == '__main__':
    main()


