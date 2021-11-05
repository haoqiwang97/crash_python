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
import torch


import numpy as np


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


"""
get_sentitivity <- function(data, columns, model){
  sentitivity <- data_frame(columns)
  sentitivity$perc_change <- 0
 
  mean_y <- mean(model$fitted.values)
  X = data %>% dplyr::select(columns)
  X = na.omit(X)
 
  for (i in 1:length(columns)){
    X_new <- X
    if(length(unique(X_new[,columns[i]])) <= 2){
      X_new[,columns[i]] <- ifelse(X_new[,columns[i]] == 1, 0, 1)
      predict_new <- predict(model, newdata = X_new, se.fit=TRUE, type='response')
      sentitivity$perc_change[i] <- (mean(predict_new$fit) - mean_y) / mean_y
    } else {
      X_new[,columns[i]] <- X_new[,columns[i]] + sd(X_new[,columns[i]])
      predict_new <- predict(model, newdata = X_new, se.fit=TRUE, type='response')
      sentitivity$perc_change[i] <- (mean(predict_new$fit) - mean_y) / mean_y
    }
  }
  return(sentitivity)
}
"""

class SensitivityNN():
    # sensitivity analysis for neural network
    def __init__(self, classifier, X, y, col_names):
        self.classifier = classifier
        self.X = X
        self.y = y
        self._y_pred()
        self.col_names = col_names
        self.sensitivity_list = np.zeros((len(col_names), y.shape[1]))
    
    def _y_pred(self):
        y_pred = np.zeros_like(self.y)
        for idx, X in enumerate(self.X):
            y_pred[idx] = self.classifier.predict(X)
        self.y_pred = y_pred
        
    def sensitivity_by_column(self, col_idx):
        # add standard deviation for numerical variables
        # switch 1 and 0 for indicator variables?
        X_new = self.X
        # select numerical variables
        X_new[:, col_idx] += torch.std(X_new[:,col_idx])

        y_pred_new = np.zeros_like(self.y_pred)
        for idx, X in enumerate(X_new):
            y_pred_new[idx] = self.classifier.predict(X)
            
        self.sensitivity_list[col_idx, :] = np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0)
        #.append(np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0))
        
    def calc_sensitivity(self):
        for i in range(2):
            self.sensitivity_by_column(i)
    
    def plot(self):
        # plot sensitivity
        pass


# temp = SensitivityNN(model, torch.from_numpy(np.array(X_val)).float(), y_val, preprocessor.get_feature_names_out())
# temp.y_pred
# temp.calc_sensitivity()
# temp.sensitivity_list
if __name__ == '__main__':
    pass
    #print_evaluation(y_train, np.zeros_like(y_train))
    #print_evaluation(y_val, np.zeros_like(y_val))

