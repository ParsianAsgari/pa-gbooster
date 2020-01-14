## About GB_Regressor:

This package provides regressing using gradient boosting technique. There are two options:

1. **XGBoost**: This is built using XGB libraries
2. **PA-GB**: This is built in part by using Scikit-learn's DecisionTreeRegressor

Once installed, the user needs to place the raw data into the ```./data/raw``` folder. Through the gateway ```gb_regressor.py```, the application will provide three functionalities: 

i. **splitdata**: The application automatically processes the data and create X_train, X_test, y_train, y_test data tables and saves them in ```./data/processed``` folder. 

ii. **train**: The application provide user options to select betweeen ```xgb``` or ```pa-gb``` regressors to train. The application will then save the serialized trained model under ```./models/``` for a provided model name.

iii. **predict**: The application provide user options to select predictor methods ```xgb``` or ```pa-gb```. NOTE: 'xgb' predictor will only with 'xgb' trained models and 'pa-gb' predictor only works with 'pa-gb' trained model. The application utilizes the splited X_test, and y_test data to perform **Predictions** and evaluate **RMSE**.


## Installation:

In the project's root directory (where this file is), run the followings:

1. Install requirements:

```bash
foo@bar pa-gbooster:$ pip install -r requirements.txt
```

2. To setup, run:

```bash
foo@bar pa-gbooster:$ pip install .
```

## How to run:

1. Place the raw .csv files into:

```bash
/data/raw
```
The program will automatically stitches the files and does the processing.

2. Navigate to the package folder ```gb_regressor```:
```bash
foo@bar pa-gbooster:$ cd gb_regressor
```

3. Run the following command to see the program's option:
```bash
foo@bar pa-gbooster/gb_regressor:$ python gb_regressor.py -h
```

Here are the options:

```bashe
GB_Regressor - Parsian Asgari's Gradient Booster Regressor

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
```

4. To run, training and testing data sets needs to be made atleast once. If you are running the training and prediction for the first time, please follow the below instructions:

4-1. **Splitting Data:**

- example: Splitting the data located in the ```./data/raw```:
```bash
foo@bar pa-gbooster/gb_regressor:$ python gb_regressor.py splitdat 0.2
```
    
4-2. **Training:** select either Parsian's Gradient Booster Regressor ```pa-ga``` or XGBoost Regressor ```xgb```:


- example: Train the ```xgb``` model, serialize and save it as ```xgb-trained-model```
```bash
foo@bar pa-gbooster/gb_regressor:$ python gb_regressor.py train xgb xgb-trained-model
```

4-3. **Predicting:** select either Parsian's Gradient Booster Predictor ```pa-ga``` or XGBoost Predictor ```xgb```:
- example: Using the step above's pre-trained serialized model, lets do some predictions on the X_test data:
```bash
foo@bar pa-gbooster/gb_regressor:$ python gb_regressor.py predict xgb xgb-trained-model
```
