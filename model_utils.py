# Standard Libraries
import os
import pickle
import json
import warnings
import sys

# Data Science Libraries: 
import torch
import scipy
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data Reprocessing 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer

# Model related imports 
import xgboost as xgb
import lightgbm as lgb
import torch.nn as nn
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
import torch.optim as optim
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# Model Evaluation & Metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# HyperParameters Tuning & Regularization
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# Others
from tabulate import tabulate
from keras.layers import Dense
from keras.models import Sequential
import csv


#Step1: Load the data from the Config file 
def load_data(file_path):
    """Load data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_excel(file_path)
    return df



def preprocess_numeric_columns(data, columns_to_convert):
    """Preprocess numeric columns by converting to numeric values."""
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors="coerce")
    return data

#using the function below to impute the hyphens and keep the NAN
def impute_hyphen_and_keep_nan(data, n_neighbors=5, weights='uniform', metric='nan_euclidean',):
    """
    Impute missing values represented by hyphen ("-") using KNNImputer while keeping original NaN values.

    Parameters:
        data (pd.DataFrame): The input DataFrame with missing values represented by hyphen ("-").
        n_neighbors (int, optional): Number of neighbors to use for KNN imputation. Default is 5.
        weights (str, optional): Weight function used in prediction. Default is 'uniform'.
        metric (str, optional): Distance metric used for missing values. Default is 'nan_euclidean'.

    Returns:
        pd.DataFrame: DataFrame with imputed missing values for hyphen ("-") and original NaN values preserved.
    """

    #Replace hyphen with np.nan
    copy = data.replace({"-": np.nan})
    data_impute = copy.drop('Cement Type', axis=1)

    #Impute missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Fit and transform the DataFrame with imputation
    imputed_data = imputer.fit_transform(data_impute)

    # Convert the imputed data back to a DataFrame
    data_imputed = pd.DataFrame(imputed_data, columns=data_impute.columns)

    # Restore the 'Cement Type' column back to the DataFrame
    data_imputed['Cement Type'] = data['Cement Type']

    #Restore the original NaN values
    data_imputed[data.isnull()] = np.nan

    return data_imputed

def read_config_file():
    with open('config_file.json') as f:
        config_file = json.load(f)
    return config_file

def load_preprocess_and_split_data(df):
    cleaned_df = df.dropna()
    X = cleaned_df.drop('28 days Compressive Strength', axis=1)
    y = cleaned_df['28 days Compressive Strength']
    return X, y


def load_and_split_data(df):
    # Load and preprocess your data here 
    X = df.drop('28 days Compressive Strength', axis=1)
    y = df['28 days Compressive Strength']
    return X, y

def evaluate_model(X, y, model, config_file, model_name):
    
    kfold = KFold(n_splits = config_file['CV_kfold'], shuffle = True, random_state = 1)
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    r2_scores = []
    train_scores = []
    test_scores = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == "XGB":
            model.fit(X_train, y_train, eval_set = [(X_test, y_test)])
        else:
            model.fit(X_train, y_train)
              

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

        mae_scores.append(mean_absolute_error(y_test, y_pred_test))
        mse_scores.append(mean_squared_error(y_test, y_pred_test))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        r2_scores.append(r2_score(y_test, y_pred_test))

    return mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores

def calculate_mean_scores(scores):
    mean_scores = [np.mean(score) for score in scores]
    return mean_scores

    
def print_evaluation_table(data):
    table = tabulate(data, headers=['Metric', 'Mean Score'], tablefmt='grid')
    print(table)
        

def train_and_evaluate_model_Linear_Regression(df, config_file):
    
    # Load data and preprocess for training
    X_train, y_train = load_preprocess_and_split_data(df)
    
    # Find the best model using a grid search
    logistic_regression_model = perform_linear_regression(X_train, y_train, config_file)
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train, logistic_regression_model, config_file,"LR")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)


def perform_linear_regression(X_train, y_train, config_file):
    # Create a pipeline for Lasso regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly_features', PolynomialFeatures()),
        ('regressor', Lasso())
    ])

    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
        'poly_features__degree': config_file['model_params']['LR']['hyperparams']['poly_features_degree'],  # Polynomial degree
        'regressor__alpha': config_file['model_params']['LR']['hyperparams']['regressor_alpha'], # Lasso alpha hyperparameter
        'regressor__fit_intercept': [True, False]
    }

    grid_search_model = GridSearchCV(pipeline, param_grid = param_grid_p, cv = config_file['model_params']['LR']['hyperparams']['kfold'], return_train_score = True)
    grid_search_model.fit(X_train, y_train)
    best_model = grid_search_model.best_estimator_
    return best_model

def train_and_evaluate_model_Decision_Tree(df, config_file):
    # Load data and split for training
    X_train, y_train = load_and_split_data(df)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Find the best hgb model using a grid search
    dt_model = perform_dt_regression(X_train, y_train, config_file)
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train, dt_model, config_file,"Decision Tree")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)

def perform_dt_regression(X_train, y_train, config_file):
    dt_model = DecisionTreeRegressor()
    
    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
         'splitter': config_file['model_params']['Decision Tree']['hyperparams']['splitter'],
         'min_samples_split': config_file['model_params']['Decision Tree']['hyperparams']['min_samples_split'],
         'min_samples_leaf': config_file['model_params']['Decision Tree']['hyperparams']['min_samples_leaf'],
         'max_features': config_file['model_params']['Decision Tree']['hyperparams']['max_features'],
         'min_impurity_decrease': config_file['model_params']['Decision Tree']['hyperparams']['min_impurity_decrease'] 
    }
    
    grid_search_model = GridSearchCV(dt_model, param_grid=param_grid_p, cv=config_file['CV_kfold'], return_train_score=True,n_jobs=-1)
    
    grid_search_model.fit(X_train, y_train)
    best_model = grid_search_model.best_estimator_
    return best_model


def train_and_evaluate_model_Random_Forest(df, config_file):
    # Load data and preprocess for training
    X_train, y_train = load_preprocess_and_split_data(df)
    
    # Find the best model using a grid search
    best_model = perform_random_forest_regression(X_train, y_train, config_file)
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train, best_model, config_file,"RF")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)
    
    # Calculate and plot feature importance for Random Forest
    calculate_and_plot_feature_importance(X_train, y_train, best_model, "Random Forest Feature Importance", "RF")

def perform_random_forest_regression(X_train, y_train, config_file):
    random_forest_model= RandomForestRegressor(random_state=1)

    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
        'n_estimators': config_file['model_params']['RF']['hyperparams']['n_estimators'],
        'max_depth': config_file['model_params']['RF']['hyperparams']['max_depth'],
        'min_samples_split': config_file['model_params']['RF']['hyperparams']['min_samples_split'],
        'min_samples_leaf': config_file['model_params']['RF']['hyperparams']['min_samples_leaf'],
        'max_features': config_file['model_params']['RF']['hyperparams']['max_features']
        
    }

    grid_search_model = GridSearchCV(random_forest_model, param_grid=param_grid_p, cv=config_file['CV_kfold'], return_train_score=True)
    grid_search_model.fit(X_train, y_train)
    best_model = grid_search_model.best_estimator_
    return best_model

def train_and_evaluate_model_XGBoost(df, config_file):
    # Load data and preprocess for training
    X_train, y_train = load_and_split_data(df)
    
    # Find the best XGBoost model using a grid search
    best_model = perform_xgboost_regression(X_train, y_train, config_file)
    
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train, best_model, config_file,"XGB")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)
    
    # Calculate and plot feature importance for XGBoost
    calculate_and_plot_feature_importance(X_train, y_train, best_model, "XGBoost Feature Importance", "XGB")



def perform_xgboost_regression(X_train_input, y_train_input, config_file):
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_input, y_train_input, test_size=0.2, random_state=1)

    xgboost_model = XGBRegressor(
        random_state=1,
        early_stopping_rounds=5, 
        eval_metric = "rmse",  
        verbose=0 
    )
    
    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
        'n_estimators': config_file['model_params']['XGB']['hyperparams']['n_estimators'],
        'max_depth': config_file['model_params']['XGB']['hyperparams']['max_depth'],
        'learning_rate': config_file['model_params']['XGB']['hyperparams']['learning_rate'],
        'min_child_weight': config_file['model_params']['XGB']['hyperparams']['min_child_weight'],
        'gamma': config_file['model_params']['XGB']['hyperparams']['gamma'],
        'colsample_bytree': config_file['model_params']['XGB']['hyperparams']['colsample_bytree']
    }
    
    # Initialize GridSearchCV
    grid_search_model = GridSearchCV(xgboost_model, param_grid=param_grid_p, cv=config_file['CV_kfold'],return_train_score=True,n_jobs=-1)
    
    #grid_search_model.fit(X_train, y_train)
    grid_search_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    best_model = grid_search_model.best_estimator_

    return best_model                    

def train_and_evaluate_model_LightGBM(df, config_file):
    # Load data and split for training
    X_train, y_train = load_and_split_data(df)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Find the best LightGBM model using a grid search
    best_model = perform_lightgbm_regression(X_train, y_train, config_file)
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train,  best_model, config_file,"LightGBM")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)
    
    # Calculate and plot feature importance for XGBoost
    calculate_and_plot_feature_importance(X_train, y_train, best_model, "LightGBM Feature Importance", "LightGBM")

def perform_lightgbm_regression(X_train, y_train, config_file):
   
    lightgbm_model = LGBMRegressor(random_state=1)
    
    # these are the evaluation sets
    X_test = X_train
    y_test = y_train
    
    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
        'n_estimators': config_file['model_params']['LightGBM']['hyperparams']['n_estimators'],
        'max_depth': config_file['model_params']['LightGBM']['hyperparams']['max_depth'],
        'learning_rate': config_file['model_params']['LightGBM']['hyperparams']['learning_rate'],
        'min_child_weight': config_file['model_params']['LightGBM']['hyperparams']['min_child_weight'],
        'colsample_bytree': config_file['model_params']['LightGBM']['hyperparams']['colsample_bytree']
    }
    

    grid_search_model = GridSearchCV(lightgbm_model, param_grid=param_grid_p, cv=config_file['CV_kfold'], return_train_score=True,n_jobs=-1)
    
    grid_search_model.fit(X_train, y_train)
    
    best_model = grid_search_model.best_estimator_
    return best_model


def train_and_evaluate_model_CatBoost(df, config_file):
    # Load data and split for training
    X_train, y_train =load_and_split_data(df)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Find the best CatBoost model using a grid search
    CatBoost_model = perform_catboost_regression(X_train, y_train, config_file)
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train,CatBoost_model, config_file,"CatBoost")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)
    
    # Calculate and plot feature importance for CatBoost
    calculate_and_plot_feature_importance(X_train, y_train, CatBoost_model, "CatBoost Feature Importance", "CatBoost")

def perform_catboost_regression(X_train, y_train, config_file):
    CatBoost_model=CatBoostRegressor(random_state=1)
    
    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
        'iterations' : config_file['model_params']['CatBoost']['hyperparams']['iterations'],
        'max_depth' : config_file['model_params']['CatBoost']['hyperparams']['max_depth'],
        'learning_rate' : config_file['model_params']['CatBoost']['hyperparams']['learning_rate']
    }

    grid_search_model = GridSearchCV(CatBoost_model, param_grid=param_grid_p, cv=config_file['CV_kfold'], return_train_score=True,n_jobs=-1)
    
    grid_search_model.fit(X_train, y_train)
    best_model = grid_search_model.best_estimator_
    return best_model


def train_and_evaluate_model_stacking(df, config_file):
    # Load data and split for training
    X_train, y_train = load_and_split_data(df)
    
    # Suppress warnings in a scoped manner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Generate a stacked model
        stacked_model = perform_stacking_regression(X_train, y_train, config_file)
        
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train, stacked_model, config_file, "stacked")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file.get('model_metrics', [])
    
    data = list(zip(metric_names, mean_scores))
    
    # Print all metrics
    print_evaluation_table(data)
    
def perform_stacking_regression(X_train, y_train, config_file):
  
    # Extract hyperparameter settings from config_file
    catboost_params = config_file['model_params']['stacked']['hyperparams']['catboost']
    lightgbm_params = config_file['model_params']['stacked']['hyperparams']['lightGBM']
    hgb_params = config_file['model_params']['stacked']['hyperparams']['Hgb']
    final_estimator_params = config_file['model_params']['stacked']['hyperparams']['final_model']['hyperparams']

    # Perform Grid Search for individual models
    CatBoost_model = GridSearchCV(CatBoostRegressor(), param_grid=catboost_params, cv=5)
    lightgbm_model = GridSearchCV(LGBMRegressor(), param_grid=lightgbm_params, cv=5)
    hgb_model = GridSearchCV(HistGradientBoostingRegressor(), param_grid=hgb_params, cv=5)

    # Fit individual models
    CatBoost_model.fit(X_train, y_train)
    lightgbm_model.fit(X_train, y_train)
    hgb_model.fit(X_train, y_train)

    # Get the best estimators from the grid search
    best_CatBoost_model = CatBoost_model.best_estimator_
    best_lightgbm_model = lightgbm_model.best_estimator_  
    best_hgb_model = hgb_model.best_estimator_

    # Create a stacking model
    stacked_model = StackingRegressor(
        estimators=[
            ('catboost', best_CatBoost_model), 
            ('lightGBM', best_lightgbm_model), 
            ('Hgb', best_hgb_model)
        ],
        final_estimator=GridSearchCV(Lasso(), param_grid=final_estimator_params, cv=5)
    )

    # Fit the stacking model
    stacked_model.fit(X_train, y_train)
    
    return stacked_model


def train_and_evaluate_model_hgb(df, config_file):
    # Load data and split for training
    X_train, y_train = load_and_split_data(df)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Find the best hgb model using a grid search
    hgb_model = perform_hgb_regression(X_train, y_train, config_file)
    
    # Evaluate the model performance using all metrics
    mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores = evaluate_model(X_train, y_train, hgb_model, config_file,"hgb")
    mean_scores = calculate_mean_scores([mae_scores, mse_scores, rmse_scores, r2_scores, train_scores, test_scores])

    metric_names = config_file['model_metrics']

    data = list(zip(metric_names, mean_scores))
    # Print all metrics
    print_evaluation_table(data)

def perform_hgb_regression(X_train, y_train, config_file):
    hgb_model = HistGradientBoostingRegressor()
    
    # Define the parameter grid for the GridSearchCV
    param_grid_p = {
        'max_iter': config_file['model_params']['hgb']['hyperparams']['max_iter'],
        'max_depth': config_file['model_params']['hgb']['hyperparams']['max_depth'],
        'learning_rate': config_file['model_params']['hgb']['hyperparams']['learning_rate'],
    }
    
    grid_search_model = GridSearchCV(hgb_model, param_grid=param_grid_p, cv=config_file['CV_kfold'], return_train_score=True,n_jobs=-1)
    
    grid_search_model.fit(X_train, y_train)
    best_model = grid_search_model.best_estimator_
    return best_model
    
def calculate_and_plot_feature_importance(X_train_input, y_train_input, model, title, model_name):
    """
    Calculate and plot feature importance using a machine learning model.

    Parameters:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.
        model: A fitted machine learning model
        title (str): Title for the plot.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test = train_test_split(X_train_input, y_train_input, test_size=0.2, random_state=1)
    
    if model_name == "XGB":
        model.fit(X_train, y_train, eval_set = [(X_test, y_test)])
    else:
        model.fit(X_train, y_train)

    feature_importance = model.feature_importances_  # Extract feature importance
    feature_names = X_train_input.columns.tolist()  # Extract feature names

    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.bar(range(len(importance_df)), importance_df['Importance'])
    plt.xticks(range(len(importance_df)), importance_df['Feature'], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def perform_xgboost_grid_search(X_train, y_train, num_folds=5):
    
    n_estimators_values = [30, 50, 100, 150, 200]
    max_depth_values = [4, 6, 8, 10, 12, 14]
    learning_rate_values = [0.01, 0.1, 0.15, 0.18, 0.2]
    subsample_values = [0.7, 0.8, 0.9]

    best_r2 = -np.inf
    best_params = {}
    
    with open('xgboost_gridsearch_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['N_Estimators', 'K_Fold', 'Max_Depth', 'Learning_Rate', 'Subsample', 'R2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                for learning_rate in learning_rate_values:
                    for subsample in subsample_values:
                        kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)
                        fold_num = 0
                        
                         # Reset the index to ensure it is integer-based and continuous
                        y_train.reset_index(drop=True, inplace=True)
                        X_train.reset_index(drop=True, inplace=True)

                        for train_idx, val_idx in kf.split(X_train):
                            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                            xgb_model = xgb.XGBRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                random_state=1
                            )
                            xgb_model.fit(X_train_fold, y_train_fold)
                            y_pred = xgb_model.predict(X_val_fold)
                            r2 = r2_score(y_val_fold, y_pred)

                            if r2 > best_r2:
                                best_r2 = r2
                                best_params = {
                                    'N_Estimators': n_estimators,
                                    'K_Fold': fold_num,
                                    'Max_Depth': max_depth,
                                    'Learning_Rate': learning_rate,
                                    'Subsample': subsample
                                }

                            writer.writerow({
                                'N_Estimators': n_estimators,
                                'K_Fold': fold_num,
                                'Max_Depth': max_depth,
                                'Learning_Rate': learning_rate,
                                'Subsample': subsample,
                                'R2': r2
                            })

                            fold_num += 1

    print("Grid search with manual cross-validation complete. Results stored in xgboost_gridsearch_results.csv")
    print(f"The best R2 score is {best_r2} with parameters {best_params}")

    return best_r2, best_params

def perform_catboost_grid_search(X_train, y_train, num_folds=5):
    iterations_values = [30, 50, 100, 150, 200]
    max_depth_values = [4, 6, 8, 10, 12, 14]
    learning_rate_values = [0.01, 0.1, 0.15, 0.18, 0.2]
    
    best_r2 = -np.inf
    best_params = {}
    
    with open('catboost_gridsearch_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Iterations', 'K_Fold', 'Max_Depth', 'Learning_Rate', 'R2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for iterations in iterations_values:
            for max_depth in max_depth_values:
                for learning_rate in learning_rate_values:
                    
                    fold_size = len(X_train) // num_folds
                    for i in range(num_folds):
                        start_idx = i * fold_size
                        end_idx = (i + 1) * fold_size
                        
                        X_train_fold = np.concatenate((X_train[:start_idx], X_train[end_idx:]))
                        y_train_fold = np.concatenate((y_train[:start_idx], y_train[end_idx:]))
                        X_val_fold = X_train[start_idx:end_idx]
                        y_val_fold = y_train[start_idx:end_idx]
                        
                        catboost_model = CatBoostRegressor(
                            iterations=iterations,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=1,
                            verbose=0
                        )
                        
                        catboost_model.fit(X_train_fold, y_train_fold)
                        y_pred = catboost_model.predict(X_val_fold)
                        r2 = r2_score(y_val_fold, y_pred)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = {
                                'Iterations': iterations,
                                'K_Fold': i,
                                'Max_Depth': max_depth,
                                'Learning_Rate': learning_rate
                            }
                        
                        writer.writerow({
                            'Iterations': iterations,
                            'K_Fold': i,
                            'Max_Depth': max_depth,
                            'Learning_Rate': learning_rate,
                            'R2': r2
                        })
                        
    print("Grid search with manual cross-validation complete. Results stored in catboost_gridsearch_results.csv")
    print(f"The best R2 score is {best_r2} with parameters {best_params}")
    
    return best_r2, best_params
