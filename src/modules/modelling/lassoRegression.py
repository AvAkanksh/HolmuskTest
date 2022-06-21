from sklearn.model_selection import train_test_split
from logs import logDecorator as lD 
import jsonref, pprint
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.modelling.lassoRegression'
linearRegression_json = jsonref.load(open('../config/modelling.json'))

@lD.log(logBase + '.dataExtraction')
def dataExtraction(logger):
    '''Extracts the dependent variables and the independent variables fromt he dataset
    
    It will read the csv file and identifies the X , y of the given dataset and returns them.
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for loggin error

    Returns
    -------
    
    X : The independent variables of the given dataset
    y : The dependent variable of the given dataset(end result which we want to predict)
    
    '''
    dataPath = linearRegression_json['linearRegression']['dataPath']
    data = pd.read_csv(dataPath)
    return data['x'].to_numpy().reshape(-1,1),data['y'].to_numpy()

# @lD.log(logBase,'.dataSplitting')
# def dataSplitting(logger,X,y):
#     '''
#     This function splits the data into train and test datasets
#     Parameters
#     ----------
#         logger : {logging.Logger}
#         The logger used for loggin error
        
#         X : independent variables from the dataset
#         y : dependent variable from the datset
        
#     Returns
#     -------
#         X_train : data of independent variables from the dataset which is used for training
#         X_test : data of independent variables from the dataset which is used for testing
#         y_train : data of dependent variables from the dataset which is used for training
#         y_test : data of dependent variables from the dataset which is used for testing
#     '''
#     X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
#     return X_train, X_test, y_train, y_test

@lD.log(logBase+'.modelBuilder')
def modelBuilder(logger):
    ''' This function helps us to build the Lasso Regression model using sklearn library

    '''
    model = linear_model.Lasso(alpha=0.1)
    return model

@lD.log(logBase + '.print_coefficients')
def print_coefficients(logger,model):
    '''
    This function helps us to print the cofficents of the trained linear Regression model for the given training dataset.
    
    Parameters
    ----------
        logger : {logging.Logger}
        The logger used for loggin error
        
        model : The model which has been trained for the given dataset.
    '''
    print("Coefficient of the trained Lasso Regression model for the given training dataset:\n",model.coef_)
    
    
@lD.log(logBase + '.plot_graphs')
def plot_graphs(logger,X_train,y_train,X_test,y_test,y_pred):
    '''
    This function helps us to visualize the scatter plot of the given data
        
    Parameters
    ----------
        logger : {logging.Logger}
        The logger used for loggin error
        X_train : data of independent variables from the dataset which is used for training
        y_train : data of dependent variables from the dataset which is used for training
        X_test : data of independent variables from the dataset which is used for testing
        y_test : The actual traget variable values of the given the test datset
        y_pred : The predicted values of the target variable using the model on the test dataset
    '''
    plt.figure()
    plt.scatter(X_train,y_train)
    plt.title("X_train vs y_train")
    plt.xlabel("X_train")
    plt.ylabel("y_train")   
    plt.savefig("../results/X_train_VS_y_train_lasso.png")
    plt.show()    
    plt.figure()
    plt.scatter(X_test,y_test)
    plt.scatter(X_test,y_pred)
    plt.legend(["y_test","y_pred"])
    plt.title("X_test vs y_test and y_pred")
    plt.xlabel("X_test")
    plt.ylabel("y")   
    plt.savefig("../results/X_test_VS_y_test_&_y_pred_lasso.png")
    plt.show()    
    
@lD.log(logBase + '.plot_graphs')
def print_MSE(logger,y_pred,y_test):
    '''
    This function helps us to print Mean Square Error on the predicts values vs actual values.
    
    Parameters
    ----------
        logger : {logging.Logger}
        The logger used for loggin error
        
        y_pred : The predicted values of the target variable using the model on the test dataset
        y_test : The actual traget variable values of the given the test datset
    '''
    print("Mean Squared Error for the given y_predicted and y_test is :",mse(y_pred,y_test))





@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for modelling
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dictionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''
    X,y = dataExtraction()
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2)
    model = modelBuilder()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_coefficients(model)
    print_MSE(y_pred, y_test)
    plot_graphs(X_train,y_train,X_test,y_test,y_pred)
    return


