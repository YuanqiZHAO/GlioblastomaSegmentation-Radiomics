# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:13:06 2020

@author: Zhiyun Gong
"""

from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from itertools import chain
import random as rd
import SimpleITK
# SimpleITK.SetGlobalDefaultCoordinateTolerance()
# from lifelines.utils import k_fold_cross_validation

clinical_pd = pd.read_csv('features/clinical.cases_selection.2020-12-05/clinical.tsv', sep='\t')
patient_ids = pd.read_csv('features/patient_ids.txt', header=None)
feature_names = list(pd.read_csv('features/feature_names.txt', header=None)[0])

# Keep 97 patients and clean data
clinical_filtered = clinical_pd.loc[clinical_pd['case_submitter_id'].isin(list(patient_ids[0]))]
survival_df_filtered = clinical_filtered.loc[:,['case_submitter_id','days_to_death','vital_status']].drop_duplicates()
survival_df_filtered.loc[survival_df_filtered.days_to_death == '\'--','days_to_death'] = 3000
survival_df_filtered['days_to_death'] = pd.to_numeric(survival_df_filtered['days_to_death'])

survival_df_filtered.loc[survival_df_filtered.vital_status=='Dead','vital_status'] = 1
survival_df_filtered.loc[survival_df_filtered.vital_status!=1,'vital_status'] = 0
survival_df_filtered['vital_status'] = pd.to_numeric(survival_df_filtered['vital_status'])

survival_df_filtered.index = range(len(survival_df_filtered))

# Yuanqi's
t1_features = pd.read_csv('features/yuanqi_features/t1_radio_features.csv', header=None)
t1Gd_features = pd.read_csv('features/yuanqi_features/t1Gd_radio_features.csv', header=None)
t2_features = pd.read_csv('features/yuanqi_features/t2_radio_features.csv', header=None)
flair_features = pd.read_csv('features/yuanqi_features/flair_radio_features.csv', header=None)

# Normalization
scaler = StandardScaler()
all_features = pd.concat([t1_features, t1Gd_features, t2_features, flair_features], axis=1)
all_features_norm =  pd.DataFrame(scaler.fit_transform(all_features))

# Feature selection
def DropLowVariance(df,events):
    ls = df.columns.tolist()
    for i in ls:
        if df.loc[events,i].var()<=0.1 or df.loc[~events, i].var()<=0.1:
            df = df.drop(i,axis=1)
    return df

def RandDropCorr(df,cutoff):
    corr = df.corr()
    l = []
    idx = []
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if abs(corr.iloc[i,j]) >cutoff and i>j:
                tup = (corr.columns[i],corr.index[j])
                index = (i,j)
                idx.append(index)
                l.append(tup)
    rd.seed(77)
    choice = []
    for i in l:
        r = rd.random()
        if i[0] not in choice and i[1] not in choice:
            if r >= 0.5:
                choice.append(i[0])
            else:
                choice.append(i[1])
    df = df.loc[:,choice]
    # df = df.drop(choice,axis=1)
    return df,choice


rfe = RFE(estimator = DecisionTreeClassifier(random_state=2), n_features_to_select = 30)
selector = rfe.fit(all_features_norm, survival_df_filtered['days_to_death'])
selected_ind = np.where(selector.support_)

all_features_selected = all_features_norm[all_features_norm.columns[selected_ind]]
events = survival_df_filtered['vital_status'].astype(bool)
all_features_drop_low_var = DropLowVariance(all_features_selected, events)

all_feature_names = [[feature_names[i]+'_' +str(j) for i in range(115)] for j in range(4)]
all_feature_names_ls = list(chain.from_iterable(all_feature_names))

all_reduced_features = [all_feature_names_ls[i] for i in list(all_features_drop_low_var.columns)]
all_features_drop_low_var.columns = all_reduced_features

all_features_drop_corr, de_corr_features = RandDropCorr(all_features_drop_low_var,0.8)
all_features_drop_corr.columns = de_corr_features

all_features_reduced = pd.concat([all_features_drop_corr,survival_df_filtered],axis=1).drop('case_submitter_id', axis=1)


my_cph = CoxPHFitter(penalizer = 0.005, l1_ratio=0.9)
# haha.drop(['original_glszm_SizeZoneNonUniformity_1'],axis=1).to_csv('truth_reg_vars.csv')
# my_cph.fit(haha.drop(['original_glszm_SizeZoneNonUniformity_1'],axis=1), duration_col = 'days_to_death', event_col='vital_status')
my_cph.fit(all_features_reduced, duration_col = 'days_to_death', event_col='vital_status')
my_cph.print_summary()