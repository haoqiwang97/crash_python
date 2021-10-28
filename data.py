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

    df_path = 'data/' + name

    df = pd.read_csv(df_path).dropna()

    df["midblock_sig"] = df["midblock_sig"].astype(np.int64)
    
    cols_int_to_float = ['transit_stops_025mi_count',
        'lane_width_ft_major', 'median_width_ft_major', 'shoulder_width_ft_major',
        'median_width_ft_minor']
    
    for i in cols_int_to_float:
        df[i] = df[i].astype(np.float64)
    
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


def load_datasets(is_small=False, is_remove_cols=True, is_classify=False, is_feat_engineering=False, is_remove_lon_lat=False):
    df = read_data(is_small=is_small, is_remove_cols=is_remove_cols, is_classify=is_classify, is_feat_engineering=is_feat_engineering, is_remove_lon_lat=is_remove_lon_lat)
    
    df_X = df.drop(columns='tot_crash_count')
    df_y = df['tot_crash_count']
    
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
    
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

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    #df = read_data(is_feat_engineering=True)
    # df2 = read_data(name='tx_crash.csv', is_remove_cols=False)
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()