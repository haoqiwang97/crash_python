# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:28:33 2021

@author: hw9335
"""

import pandas as pd


df = pd.read_csv("data/ints_osm_appch_adt_2019.csv")

df = df[['int_id', 'major', 'adt_adj', 'hy_1']].rename(columns = {'adt_adj': 'adt_2019', 'hy_1': 'adt_2018'})
df['int_id'] = df['int_id'].astype(str) + df['major']

df.head()

df['diff'] = df['adt_2019'] - df['adt_2018']
df.head()
df['diff'].describe()
df['diff'].hist()
