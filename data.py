# import pyreadr
#
# df = pyreadr.read_r('data/IntDataV1.RDS')
#
# df1 = df[None]
# df2 = df1.T

import pandas as pd
import numpy as np


def read_data(name='IntDataV1.csv', is_small=True, is_remove_cols=True):
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

    if is_remove_cols:
        # df = pd.read_csv(df_path, usecols=cols_keep).dropna()
        df = pd.read_csv(df_path).dropna().drop(columns=cols_drop)
    else:
        df = pd.read_csv(df_path).dropna()

    df['crash'] = np.where(df['tot_crash_count'] == 0, 0, 1)

    if is_small:
        df = df.sample(frac=0.1, random_state=1)

    return df


if __name__ == '__main__':
    df = read_data()
    #df2 = read_data(name='tx_crash.csv', is_remove_cols=False)
