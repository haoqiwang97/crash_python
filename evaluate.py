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
import matplotlib.pyplot as plt
import pandas as pd

def print_evaluation(y_val, y_val_pred):
    result_dict = {
        'rmse': mean_squared_error(y_val, y_val_pred, squared=False),
        'r2_score': r2_score(y_val, y_val_pred),
        'mean_absolute_error': mean_absolute_error(y_val, y_val_pred),
        #'mean_absolute_percentage_error': mean_absolute_percentage_error(y_val, y_val_pred),
        'median_absolute_error': median_absolute_error(y_val, y_val_pred),
        'explained_variance_score': explained_variance_score(y_val, y_val_pred)
    }
    
    for key, value in result_dict.items():
        result_dict[key] = round(result_dict[key], 3)
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
    for key, value in result_dict.items():
        result_dict[key] = round(result_dict[key], 3)
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

def custom_replace(tensor, on_zero=1, on_non_zero=0):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res


class SensitivityNN():
    # sensitivity analysis for neural network
    def __init__(self, classifier, X, y, x_col_names, y_col_names):
        self.classifier = classifier
        self.X = X
        self.y = y
        #self.y_pred = self._y_pred(X, y)
        self.y_pred = self.classifier.predict(X)
        self.x_col_names = x_col_names
        self.y_col_names = y_col_names
        self._features_list()
        # self.sensitivity_list = np.zeros((self.n_features, y.shape[1]))
        self.sensitivity_list = []
        self.get_plot_label_names()
        
    def _y_pred(self, X, y):
        y_pred = np.zeros_like(y)
        for idx, X in enumerate(X):
            y_pred[idx] = self.classifier.predict(X)
        return y_pred
        
    def _features_list(self):
        num_features_list = []
        cat_features_list = []
        x_col_names_compact =[]
        
        for idx, value in enumerate(self.x_col_names):
            if value.startswith('num'):
                num_features_list.append(idx)
                x_col_names_compact.append(value)
            else:
                cat_features_list.append(idx)
                if idx % 2 == 0:
                    x_col_names_compact.append(value)
        
        n_features = int(len(num_features_list) + len(cat_features_list)/2)
        self.num_features_list = num_features_list
        self.cat_features_list = cat_features_list
        self.n_features = n_features
        self.x_col_names_compact = x_col_names_compact
        
    def sensitivity_by_column_num(self, col_idx):
        # add standard deviation for numerical variables
        # switch 1 and 0 for indicator variables?
        X_new = self.X.clone()
        # select numerical variables
        X_new[:, col_idx] += torch.std(X_new[:,col_idx])

        y_pred_new = self.classifier.predict(X_new)
        
        # self.sensitivity_list[col_idx, :] = np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0)
        self.sensitivity_list.append(np.mean((y_pred_new.numpy() - self.y_pred.numpy()) / self.y_pred.numpy(), axis=0))
        #.append(np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0))


    def sensitivity_by_column_num2(self, col_idx):
        # another way of calculate sensitivity
        X_new = self.X.clone()
        # select numerical variables
        X_new[:, col_idx] += torch.std(X_new[:,col_idx])
        y_pred_new = self.classifier.predict(X_new)
        
        # self.sensitivity_list[col_idx, :] = np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0)
        self.sensitivity_list.append((np.mean(y_pred_new.numpy()) - np.mean(self.y_pred.numpy())) / np.mean(self.y_pred.numpy(), axis=0))


    def sensitivity_by_column_cat(self, col_idx):
        X_new = self.X.clone()
        X_new[:, col_idx] = custom_replace(X_new[:, col_idx])
        X_new[:, col_idx+1] = custom_replace(X_new[:, col_idx+1])
        
        y_pred_new = self.classifier.predict(X_new)
        # self.sensitivity_list[col_idx, :] = np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0)
        self.sensitivity_list.append(np.mean((y_pred_new.numpy() - self.y_pred.numpy()) / self.y_pred.numpy(), axis=0))


    def sensitivity_by_column_cat2(self, col_idx):
        # another way of calculate sensitivity
        X_new = self.X.clone()
        X_new[:, col_idx] = custom_replace(X_new[:, col_idx])
        X_new[:, col_idx+1] = custom_replace(X_new[:, col_idx+1])
        
        y_pred_new = self.classifier.predict(X_new)
        # self.sensitivity_list[col_idx, :] = np.mean((y_pred_new - self.y_pred) / self.y_pred, axis=0)
        self.sensitivity_list.append((np.mean(y_pred_new.numpy()) - np.mean(self.y_pred.numpy())) / np.mean(self.y_pred.numpy(), axis=0))

        
    def calc_sensitivity(self):
        for i in self.num_features_list:
            self.sensitivity_by_column_num(i)
            print(i, self.x_col_names[i], "--sensitivity calculated!")
        
        for i in self.cat_features_list[::2]:
            self.sensitivity_by_column_cat(i)
            print(i, self.x_col_names[i], "--sensitivity calculated!")


    def calc_sensitivity2(self):
        # another way of calculate sensitivity
        for i in self.num_features_list:
            self.sensitivity_by_column_num2(i)
            print(i, self.x_col_names[i], "--sensitivity calculated!")
        
        for i in self.cat_features_list[::2]:
            self.sensitivity_by_column_cat2(i)
            print(i, self.x_col_names[i], "--sensitivity calculated!")

    def get_plot_label_names(self):
        plot_label_names = []
        for value1 in self.x_col_names_compact:
            for value2 in self.y_col_names:
                plot_label_names.append(value1 + ": " + value2)
        self.plot_label_names = plot_label_names

    def plot(self, is_save_figure=False, figure_name=None, plot6=False):
        # plot sensitivity
        # plot6: one severity on one plot
        
        plot_label_names = self.plot_label_names
        
        plot_values = np.array(self.sensitivity_list)
        
        if not plot6:
            fig, ax = plt.subplots(figsize=[12, 40])
            
            ax.barh(plot_label_names, plot_values.flatten())
            ax.set(#xlim=(0, 10), ylim=(-2, 2),
                   #xlabel='x', ylabel='sin(x)',
                   title='Sensitivity analysis');
            
            plt.tight_layout()
            
            if is_save_figure:
                # name = figure_name + '.pdf'
                name = figure_name + '.png'
                fig.savefig(name, dpi=300)
        else:
            figs = []
            outcome_len = len(self.y_col_names)
            #pdb.set_trace()
            for i in range(outcome_len):
                fig, ax = plt.subplots(figsize=[12, 10])
                ax.barh(self.x_col_names_compact, plot_values.flatten()[i::6])
                title_text = 'Sensitivity analysis: ' + self.y_col_names[i]
                ax.set(#xlim=(0, 10), ylim=(-2, 2),
                       #xlabel='x', ylabel='sin(x)',
                       title=title_text);
                
                plt.tight_layout()
                figs.append(fig)
                
                if is_save_figure:
                    for i in range(len(figs)):
                        # name = figure_name + '_' + self.y_col_names[i] + '.pdf'
                        name = figure_name + '_' + self.y_col_names[i] + '.png'
                        figs[i].savefig(name, dpi=300)

    def spreadsheet(self, is_save_spreadsheet=False, spreadsheet_name=None):
        if len(self.y_col_names) == 1:
            spreadsheet = pd.DataFrame(list(zip(self.plot_label_names, np.squeeze(np.array(self.sensitivity_list)))))
        else:
            spreadsheet = pd.DataFrame(list(zip(self.plot_label_names, np.array(self.sensitivity_list).reshape(-1))))
            
        name = spreadsheet_name + '.csv'
        spreadsheet.to_csv(name, header=False, index=False)
            
    def save_report(self, is_save=True, name=None, plot6=False):
        self.plot(is_save_figure=is_save, figure_name=name, plot6=plot6)
        self.spreadsheet(is_save_spreadsheet=is_save, spreadsheet_name=name)


if __name__ == '__main__':
    pass
    
    # project1
    sens = SensitivityNN(classifier=model, 
                          X=torch.from_numpy(np.array(X_val)).float(), 
                          y=y_val, 
                          x_col_names=preprocessor.get_feature_names_out(), 
                          y_col_names=y_col_names)     
    sens.calc_sensitivity2()
    # sens.plot(is_save_figure=True, figure_name="project1/val_sens2")
    # sens.spreadsheet(is_save_spreadsheet=True, spreadsheet_name='project1/val_sens2')
    sens.save_report(is_save=True, name="project1/val_sens2", plot6=False)
    import pickle
    with open('project1/val_sens2.pickle', 'wb') as handle:
        pickle.dump(sens, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open('project1/val_sens2.pickle', 'rb') as handle:
    #     b = pickle.load(handle)    
    
    
    
    # project2
    sens = SensitivityNN(classifier=model, 
                          X=torch.from_numpy(np.array(X_val)).float(), 
                          y=y_val, 
                          x_col_names=preprocessor.get_feature_names_out(), 
                          y_col_names=y_col_names)     
    sens.calc_sensitivity2()
    # sens.plot(is_save_figure=True, figure_name="project2/val_sens2")
    # sens.spreadsheet(is_save_spreadsheet=True, spreadsheet_name='project2/val_sens2')
    sens.save_report(is_save=True, name="project2/val_sens2", plot6=True)
    sens.save_report(is_save=True, name="project2/val_sens2", plot6=False)
    import pickle
    with open('project2/val_sens2.pickle', 'wb') as handle:
        pickle.dump(sens, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open('project1/val_sens2.pickle', 'rb') as handle:
    #     b = pickle.load(handle)    
    len(sens.plot_label_names)
    np.array(sens.sensitivity_list).reshape(-1)
