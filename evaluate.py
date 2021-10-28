# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:11:52 2021

@author: hw9335
"""

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error


def print_evaluation(y_val, y_val_pred):
    result_dict = {
        'rmse': mean_squared_error(y_val, y_val_pred, squared=False),
        'r2_score': r2_score(y_val, y_val_pred),
        'mean_absolute_error': mean_absolute_error(y_val, y_val_pred),
        'mean_absolute_percentage_error': mean_absolute_percentage_error(y_val, y_val_pred),
        'median_absolute_error': median_absolute_error(y_val, y_val_pred),
        'explained_variance_score': explained_variance_score(y_val, y_val_pred)
    }
    return result_dict


def print_evaluation_all(y_val, y_val_pred, y_train, y_train_pred):
    result_dict = {
        'rmse_train': mean_squared_error(y_train, y_train_pred, squared=False),
        'rmse_val': mean_squared_error(y_val, y_val_pred, squared=False),
        'r2_score_train': r2_score(y_train, y_train_pred),
        'r2_score_val': r2_score(y_val, y_val_pred),
        'mean_absolute_error_train': mean_absolute_error(y_train, y_train_pred),
        'mean_absolute_error_val': mean_absolute_error(y_val, y_val_pred),
        'mean_absolute_percentage_error_train': mean_absolute_percentage_error(y_train, y_train_pred),
        'mean_absolute_percentage_error_val': mean_absolute_percentage_error(y_val, y_val_pred),
        'median_absolute_error_train': median_absolute_error(y_train, y_train_pred),
        'median_absolute_error_val': median_absolute_error(y_val, y_val_pred),
        'explained_variance_score_train': explained_variance_score(y_train, y_train_pred),
        'explained_variance_score_val': explained_variance_score(y_val, y_val_pred)
    }
    return result_dict


if __name__ == '__main__':
    pass
    #print_evaluation(y_train, np.zeros_like(y_train))
    #print_evaluation(y_val, np.zeros_like(y_val))
