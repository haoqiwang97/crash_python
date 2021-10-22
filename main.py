import argparse
from data import *
from evaluate import *

def _parse_args():
    pass


X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()
print("%i train exs, %i dev exs, %i test exs" % (len(y_train), len(y_val), len(y_test)))

from sklearn.neural_network import MLPRegressor


regr = MLPRegressor(random_state=1, hidden_layer_sizes=(32, 32))
regr.fit(X_train, y_train)
print_evaluation_all(y_val, regr.predict(X_val), y_train, regr.predict(X_train))
"""
{'rmse_train': 8.553915194554236,
 'rmse_val': 12.080965850866708,
 'r2_score_train': 0.6960283159879984,
 'r2_score_val': 0.42546866914412207,
 'mean_absolute_error_train': 2.4911227590600893,
 'mean_absolute_error_val': 2.7781083526912753,
 'mean_absolute_percentage_error_train': 3262853998109949.5,
 'mean_absolute_percentage_error_val': 3456464339849024.0,
 'median_absolute_error_train': 0.3542516706761658,
 'median_absolute_error_val': 0.3574926208790601,
 'explained_variance_score_train': 0.6961102419109616,
 'explained_variance_score_val': 0.42561490845825445}
"""

X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(is_remove_lon_lat=True)
regr = MLPRegressor(random_state=1, hidden_layer_sizes=(5))
regr.fit(X_train, y_train)
print_evaluation_all(y_val, regr.predict(X_val), y_train, regr.predict(X_train))
"""
{'rmse_train': 10.815070874719089,
 'rmse_val': 11.553563890343979,
 'r2_score_train': 0.5140831170073712,
 'r2_score_val': 0.4745367541493415,
 'mean_absolute_error_train': 2.8799263228282603,
 'mean_absolute_error_val': 2.9495983510488664,
 'mean_absolute_percentage_error_train': 4074336238771302.5,
 'mean_absolute_percentage_error_val': 4218080327931728.0,
 'median_absolute_error_train': 0.6629017912484265,
 'median_absolute_error_val': 0.6650102826496709,
 'explained_variance_score_train': 0.5144109184899529,
 'explained_variance_score_val': 0.47485922831677774}
"""