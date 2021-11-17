# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:07:23 2021

@author: hw9335
"""

import pandas as pd
import seaborn as sns


df = pd.read_csv("data/df_model2.csv")

# look at correlations beteen severities and total crash count
temp = df[['tot_crash_count', 'sev_small', 'sev_incapac', 'sev_nonincapac', 'sev_possible', 'sev_killed']]
temp2 = temp.corr()

sns.heatmap(temp2, annot=True)
sns.pairplot(temp)
