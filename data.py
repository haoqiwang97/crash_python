# import pyreadr
#
# df = pyreadr.read_r('data/IntDataV1.RDS')
#
# df1 = df[None]
# df2 = df1.T

import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split

import pickle


def read_raw(name='IntDataV1.csv'):
    df_path = 'data/' + name

    df = pd.read_csv(df_path).dropna()

    df["midblock_sig"] = df["midblock_sig"].astype(np.int64)
    df["approaches"] = df["approaches"].astype(np.float64)
    df["lanes_major"] = df["lanes_major"].astype(np.float64)
    
    cols_int_to_float = ['transit_stops_025mi_count',
        'lane_width_ft_major', 'median_width_ft_major', 'shoulder_width_ft_major',
        'median_width_ft_minor']
    
    for i in cols_int_to_float:
        df[i] = df[i].astype(np.float64)
    return df


def read_data(name='IntDataV1.csv', is_small=True, is_remove_cols=True, is_classify=True, is_feat_engineering=False, is_remove_lon_lat=False):
    """
    Read all data
    correct column types, drop missing data
    remove junction, center, ped_crash_count, Austin, ped_crash_count_fatal columns
    cols_all = ['int_id', 'signal', 'junction', 'midblock_sig', 'center', 'descr', 'lat', 'lon', 'Austin',
            'ped_crash_count', 'ped_crash_count_fatal', 'signalized_ind', 'approaches', 'on_system',
            'dist_near_school_mi', 'dist_near_hops_mi', 'transit_ind', 'transit_stops_025mi_count',
            'sidewalk_lenght_150ft_ft', 'aadt_lane_major', 'aadt_lane_minor', 'a_rural', 'a_small_urban',
            'a_urbanized',
            'a_large_urban', 'DVMT_major', 'log_DVMT_major', 'lanes_major', 'lane_width_ft_major', 'median_major',
            'median_width_ft_major', 'should_major', 'shoulder_width_ft_major', 'truck_perc_major', 'f_local_major',
            'f_collector_major', 'f_arterial_major', 'f_unknown_major', 'd_1way_major', 'd_2way_undiv_major',
            'd_2way_divid_major', 'DVMT_minor', 'log_DVMT_minor', 'lanes_minor', 'lane_width_ft_minor',
            'median_minor',
            'median_width_ft_minor', 'should_minor', 'truck_perc_minor', 'f_local_minor', 'shoulder_width_ft_minor',
            'f_collector_minor', 'f_arterial_minor', 'f_unknown_minor', 'd_1way_minor', 'd_2way_undiv_minor',
            'd_2way_divid_minor', 'speed_lim_mph_major', 'speed_lim_mph_minor', 'tot_WMT', 'tot_WMT_pop',
            'tot_WMT_sqmi', 'log_tot_WMT', 'log_tot_WMT_pop', 'log_tot_WMT_sqmi', 'tot_crash_count']
    :return:
    """
    cols_drop = ['int_id', 'descr', 'junction', 'center',
                 'ped_crash_count', 'Austin', 'ped_crash_count_fatal', 'signal', 'transit_ind', 'median_major', 'should_major', 'median_minor', 'should_minor']

    cols_log = ['log_DVMT_major',
                'log_DVMT_minor',
                'log_tot_WMT',
                'log_tot_WMT_pop',
                'log_tot_WMT_sqmi']  # df.columns[df.columns.str.startswith('log')]

    cols_drop.extend(cols_log)
    # cols_keep = list(set(cols_all) - set(cols_drop))

    df = read_raw(name=name)
    
    if is_remove_cols:
        # df = pd.read_csv(df_path, usecols=cols_keep).dropna()
        df = df.drop(columns=cols_drop)
    
    if is_classify:
        df['crash'] = np.where(df['tot_crash_count'] == 0, 0, 1)
    
    if is_feat_engineering:
        """
        cols_keep = ['aadt_lane_major', 'DVMT_major', 'aadt_lane_minor',
                     'speed_lim_mph_major', 'f_local_major', 'DVMT_minor',
                     'f_arterial_major', 'lanes_major', 'tot_WMT_sqmi', 'signalized_ind',
                     'lane_width_ft_major', 'speed_lim_mph_minor', 'f_local_minor',
                     'truck_perc_major', 'tot_WMT', 'approaches', 'tot_WMT_pop', 'lon',
                     'lane_width_ft_minor', 'lat',
                     'tot_crash_count']
        """
        cols_keep = ['aadt_lane_major', 'DVMT_major', 'aadt_lane_minor',
                     'speed_lim_mph_major', 'DVMT_minor',
                     'lanes_major', 'tot_WMT_sqmi', 'signalized_ind',
                     'lane_width_ft_major', 'speed_lim_mph_minor',
                     'truck_perc_major', 'tot_WMT', 'approaches', 'tot_WMT_pop', 'lon',
                     'lane_width_ft_minor', 'lat',
                     # 'dist_near_hops_mi', 'dist_near_school_mi', # not useful
                     'tot_crash_count']
        df = df[cols_keep]
    
    if is_remove_lon_lat:
        df = df.drop(columns=['lat', 'lon'])
        
    if is_small:
        df = df.sample(frac=0.1, random_state=1)

    return df


def transform_data_log1p(df, is_transform_y=True):
    cols_log1p = ['aadt_lane_major', 'DVMT_major', 'aadt_lane_minor',
                  'DVMT_minor', 'tot_WMT_sqmi', 'truck_perc_major', 'tot_WMT',
                  'tot_WMT_pop',
                  'tot_crash_count']
    
    if is_transform_y == False:
        cols_log1p.remove('tot_crash_count')
        
    for i in cols_log1p:
        df[i] = np.log1p(df[i])
    
    return df


def transform_data_ind(X):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_include="float64")),
        ('cat', categorical_transformer, selector(dtype_include="int64"))
        ])
    
    preprocessor.fit(X)
    
    X = preprocessor.transform(X)
    return X, preprocessor

"""    
def transform_data_nn(X_train, X_val, X_test):
    # transform data for neural network
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_include="float64")),
        ('cat', categorical_transformer, selector(dtype_include="int64"))
        ])
    
    preprocessor.fit(X_train)
    
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    return X_train, X_val, X_test, preprocessor


def load_datasets(is_small=False, is_remove_cols=True, is_classify=False, is_feat_engineering=False, is_remove_lon_lat=False):
    df = read_data(is_small=is_small, is_remove_cols=is_remove_cols, is_classify=is_classify, is_feat_engineering=is_feat_engineering, is_remove_lon_lat=is_remove_lon_lat)
    
    df_X = df.drop(columns='tot_crash_count')
    df_y = df['tot_crash_count']
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
    
    X_train, X_val, X_test, preprocessor = transform_data_nn(X_train, X_val, X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
"""

def transform_data_nn(X_train, X_val):
    # transform data for neural network
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_include="float64")),
        ('cat', categorical_transformer, selector(dtype_include="int64"))
        ])
    
    preprocessor.fit(X_train)
    
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)
    return X_train, X_val, preprocessor


def load_datasets(is_small=False, is_remove_cols=True, is_classify=False, is_feat_engineering=False, is_remove_lon_lat=False):
    df = read_data(is_small=is_small, is_remove_cols=is_remove_cols, is_classify=is_classify, is_feat_engineering=is_feat_engineering, is_remove_lon_lat=is_remove_lon_lat)
    
    df_X = df.drop(columns='tot_crash_count')
    df_y = df['tot_crash_count']
    
    X_train, X_val, y_train, y_val = train_test_split(df_X, df_y, test_size=0.2, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    X_train, X_val, preprocessor = transform_data_nn(X_train, X_val)
    
    return X_train, y_train, X_val, y_val, preprocessor


def read_new_y(path="data/crash_int_severities_years.csv"):
    # read y by severities
    new_y = pd.read_csv(path)
    new_y['sev_small'] = new_y['sev_notinjured'] + new_y['sev_unknown']
    return new_y
    

def load_datasets_severities_sum():
    df = read_raw()
    
    # drop some columns
    cols_drop = ['descr', 'junction', 'center',
                 'ped_crash_count', 
                 'Austin', 'ped_crash_count_fatal', 'signal', 'transit_ind', 'median_major', 'should_major', 'median_minor', 'should_minor',
                 'lon', 'lat']
    cols_log = ['log_DVMT_major',
                'log_DVMT_minor',
                'log_tot_WMT',
                'log_tot_WMT_pop',
                'log_tot_WMT_sqmi']
    cols_drop.extend(cols_log)
    df = df.drop(columns=cols_drop)
    
    new_y = read_new_y()
    new_y = new_y.groupby(['int_id']).agg({'sev_small': 'sum',
                                           'sev_incapac': 'sum',
                                           'sev_nonincapac': 'sum',
                                           'sev_possible': 'sum',
                                           'sev_killed': 'sum'})
    
    # merge data
    df = pd.merge(df, new_y, how='left', on='int_id')
    df = df.fillna(0)
    df = df.drop(columns = ['int_id'])
    
    df_X = df.drop(columns=['sev_small', 'sev_incapac', 'sev_nonincapac', 'sev_possible', 'sev_killed', 'tot_crash_count'])
    
    y_col_names = ['sev_small', 'sev_incapac', 'sev_nonincapac', 'sev_possible', 'sev_killed', 'tot_crash_count']
    df_y = df[y_col_names]
    # split data
    X_train, X_val, y_train, y_val = train_test_split(df_X, df_y, test_size=0.2, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1)
    
    #X_train, X_val, X_test, preprocessor = transform_data_nn(X_train, X_val, X_test)
    X_train, X_val, preprocessor = transform_data_nn(X_train, X_val)
    return X_train, y_train, X_val, y_val, preprocessor, y_col_names


class CrashExample:
    x_temporal_row_names = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    x_temporal_col_names = ['sev_small', 'sev_incapac', 'sev_nonincapac', 'sev_possible', 'sev_killed', 'tot_crash_count']
    geo_col_names = ['num__approaches', 'num__dist_near_school_mi',
                     'num__dist_near_hops_mi', 'num__transit_stops_025mi_count',
                     'num__sidewalk_lenght_150ft_ft', 'num__aadt_lane_major',
                     'num__aadt_lane_minor', 'num__DVMT_major', 'num__lanes_major',
                     'num__lane_width_ft_major', 'num__median_width_ft_major',
                     'num__shoulder_width_ft_major', 'num__truck_perc_major',
                     'num__DVMT_minor', 'num__lanes_minor', 'num__lane_width_ft_minor',
                     'num__median_width_ft_minor', 'num__truck_perc_minor',
                     'num__shoulder_width_ft_minor', 'num__f_unknown_minor',
                     'num__speed_lim_mph_major', 'num__speed_lim_mph_minor',
                     'num__tot_WMT', 'num__tot_WMT_pop', 'num__tot_WMT_sqmi',
                     'cat__midblock_sig_0', 'cat__midblock_sig_1',
                     'cat__signalized_ind_0', 'cat__signalized_ind_1',
                     'cat__on_system_0', 'cat__on_system_1', 'cat__a_rural_0',
                     'cat__a_rural_1', 'cat__a_small_urban_0', 'cat__a_small_urban_1',
                     'cat__a_urbanized_0', 'cat__a_urbanized_1', 'cat__a_large_urban_0',
                     'cat__a_large_urban_1', 'cat__f_local_major_0',
                     'cat__f_local_major_1', 'cat__f_collector_major_0',
                     'cat__f_collector_major_1', 'cat__f_arterial_major_0',
                     'cat__f_arterial_major_1', 'cat__f_unknown_major_0',
                     'cat__f_unknown_major_1', 'cat__d_1way_major_0',
                     'cat__d_1way_major_1', 'cat__d_2way_undiv_major_0',
                     'cat__d_2way_undiv_major_1', 'cat__d_2way_divid_major_0',
                     'cat__d_2way_divid_major_1', 'cat__f_local_minor_0',
                     'cat__f_local_minor_1', 'cat__f_collector_minor_0',
                     'cat__f_collector_minor_1', 'cat__f_arterial_minor_0',
                     'cat__f_arterial_minor_1', 'cat__d_1way_minor_0',
                     'cat__d_1way_minor_1', 'cat__d_2way_undiv_minor_0',
                     'cat__d_2way_undiv_minor_1', 'cat__d_2way_divid_minor_0',
                     'cat__d_2way_divid_minor_1']
    y_row_name = [2019]
    
    
    def __init__(self, int_id, x_temporal, x_geo, y):
        self.int_id = int_id
        self.x_temporal = x_temporal
        self.x_geo = x_geo
        self.y = y
        
        
    def __repr__(self):
        return "x_temporal=" + repr(self.x_temporal) + "\n x_geo=" + repr(self.x_geo) + "\n y=" + repr(self.y)


    def __str__(self):
        return self.__repr__()    
    
    
def load_datasets_severities_ind(first_time=False):
    if first_time:
        new_y = read_new_y()
        df = read_raw()
        
        y_col_names = ['int_id', 'year', 'sev_small', 'sev_incapac', 'sev_nonincapac', 'sev_possible', 'sev_killed']
        new_y = new_y[y_col_names]
        year = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019] # input 2010 - 2018, predict 2019
        
        # new_y['int_id'].nunique() = 381513
        # df['int_id'].nunique() = 707152
        
        # use dict
        new_y_grouped_dict = {}
        new_y_grouped = new_y.groupby('int_id')
        for key, item in new_y_grouped:
            new_y_grouped_dict[key] = new_y_grouped.get_group(key).set_index('year').drop(columns=['int_id'])
        
        # pad 0s
        primal = pd.DataFrame(data=0, index=year, columns=y_col_names, dtype=np.int8)
        new_y_grouped_dict_pad = dict.fromkeys(df['int_id']) 
        
        # TODO: check input in cs388
        
        for key in df['int_id']:
            new_y_grouped_dict_pad[key] = primal.copy() # set 0 first
            if key in new_y_grouped_dict:
                new_y_grouped_dict_pad[key].update(new_y_grouped_dict[key])
        
        for key, value in new_y_grouped_dict_pad.items():
            new_y_grouped_dict_pad[key] = new_y_grouped_dict_pad[key].to_numpy()
            
        new_y_ind = {'row_names': year,
                     'col_names': y_col_names,
                     'data': new_y_grouped_dict_pad}
        
        with open('data/y_mdl3.pickle', 'wb') as handle:
            pickle.dump(new_y_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    else:
        with open('data/y_mdl3.pickle', 'rb') as handle:
            new_y_ind = pickle.load(handle)
    
    df = read_raw()
    
    # drop some columns
    cols_drop = ['descr', 'junction', 'center',
                 'ped_crash_count', 
                 'Austin', 'ped_crash_count_fatal', 'signal', 'transit_ind', 'median_major', 'should_major', 'median_minor', 'should_minor',
                 'lon', 'lat',
                 'tot_crash_count']
    cols_log = ['log_DVMT_major',
                'log_DVMT_minor',
                'log_tot_WMT',
                'log_tot_WMT_pop',
                'log_tot_WMT_sqmi']
    cols_drop.extend(cols_log)
    df = df.drop(columns=cols_drop)
    int_id = list(df.pop('int_id'))
    X, preprocessor = transform_data_ind(df)
    
    #df.columns
    exs = []
    for i in range(len(int_id)):
    # for i in range(1000):
        a = new_y_ind['data'][int_id[i]]
        b = new_y_ind['data'][int_id[i]].sum(axis=1)[:, np.newaxis]
        new_y_ind_tot = np.hstack((a, b))
        exs.append(
            CrashExample(
                int_id[i], 
                #new_y_ind['data'][int_id[i]][:9, :], #x_temporal, 
                new_y_ind_tot[:9, :],
                X[i], 
                #new_y_ind['data'][int_id[i]][9, :]#y
                new_y_ind_tot[9, :]
                ))
        
        if i % 100000 == 0:
            print("example: ", i)
    train_exs, test_exs = train_test_split(exs, test_size=0.2, random_state=1)
    return train_exs, test_exs


# visualize model
def visualize_model(model, y_pred):
    import os
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz\\bin'
    from torchviz import make_dot
    
    res = make_dot(y_pred, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    return res
    
    
if __name__ == '__main__':
    #df = read_data(is_feat_engineering=True)
    # df2 = read_data(name='tx_crash.csv', is_remove_cols=False)
    #X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()
    #load_datasets_severities_sum()
    train_exs, test_exs = load_datasets_severities_ind()
    visualize_model(model, model(torch.from_numpy(np.array(X_val)).float()))

    visualize_model(model, model(torch.from_numpy(test_exs[1].x_temporal).float(), torch.from_numpy(test_exs[1].x_geo).float()))
