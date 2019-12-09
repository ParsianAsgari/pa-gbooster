# Huami Take Home Assignment

Please use Python only for the assignment:

### The data:

Given the data for the regression problem, 15 records for 15 patients:

https://drive.google.com/file/d/11JVsckOQ741y7tpPMdSvAAhMFYtnAmdg/view?usp=sharing

It includes 66 columns.

Target - 'target' (1st column)

All features are representing parameters of the human body (e.g. heart rate/respiration, etc.); they are numerical.

### The tasks:

1. Prepare and train regression model with one of boosting algorithms (e.g. lightGBM, XGBoost). Please, use one-hold-out cross-validation (patient-wise) only. 

2. Create the class, which includes methods:
a. method for loading the model and storing it in the memory
b. method for loading the data, apply all required preprocessing, and making predictions. 

For task #2.b, please, implement the prediction function (forward pass) for Boosting without using XGboost or LightGBM libraries (low-level implementation).

Please provide a solution in a form that is suitable for running (i.e. by giving requirements.txt for pip/virtualenv or as a Docker image with the environment set up).


## About GB_Booster:

This package provides regressing using gradient boosting. There are two methods:

1. **XGBoost**: This is built using XGB libraries
2. **PA-GB**: This is built in part by using Scikit-learn's DecisionTreeRegressor

Once installed, the user needs to place the raw data into the ```./data/raw``` folder. Through the gateway ```gb_booster.py```, the application will provide three functionalities: 

i. **splitdata**: The application automatically processes the data and create X_train, X_test, y_train, y_test data tables and saves them in ```./data/processed``` folder. 

ii. **train**: The application provide user options to select betweeen ```xgb``` or ```pa-gb``` regressors to train. The application will then save the serialized trained model under ```./models/``` for a provided model name.

iii. **predict**: The application provide user options to select predictor methods ```xgb``` or ```pa-gb```. NOTE: 'xgb' predictor will only with 'xgb' trained models and 'pa-gb' predictor only works with 'pa-gb' trained model. The application utilizes the splited X_test, and y_test data to perform **Predictions** and evaluate **RMSE**.


## Installation:

In the project's root directory (where this file is), run the followings:

1. Install requirements:

```bash
foo@bar huami-interview:$ pip install -r requirements.txt
```

2. To setup, run:

```bash
foo@bar huami-interview:$ pip install .
```

## How to run:

1. Place the raw .csv files into:

```bash
/data/raw
```
The program will automatically stitches the files and does the processing.

2. Navigate to the package folder ```gb_booster```:
```bash
foo@bar huami-interview:$ cd gb_booster
```

3. Run the following command to see the program's option:
```bash
foo@bar huami-interview/gb_booster:$ python gb_booster.py -h
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
foo@bar huami-interview/gb_booster:$ python splitdat 0.2
```
    
4-2. **Training:** select either Parsian's Gradient Booster Regressor ```pa-ga``` or XGBoost Regressor ```xgb```:


- example: Train the ```xgb``` model, serialize and save it as ```xgb-trained-model```
```bash
foo@bar huami-interview/gb_booster:$ python gb_booster.py train xgb xgb-trained-model
```

4-3. **Predicting:** select either Parsian's Gradient Booster Predictor ```pa-ga``` or XGBoost Predictor ```xgb```:
- example: Using the step above's pre-trained serialized model, lets do some predictions on the X_test data:
```bash
foo@bar huami-interview/gb_booster:$ python gb_booster.py predict xgb xgb-trained-model
```