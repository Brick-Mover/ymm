#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:27:29 2018

@author: kgicmd
"""
import os
os.chdir("/Users/kgicmd/Downloads/ymm")

import numpy as np
import pandas as pd

import multiprocessing as mp
from tqdm import tqdm

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

from sklearn.metrics import mean_squared_error

def read_data(file_name):
    """
    read csv files
    """
    data = pd.read_csv(file_name)
    return data

# simple stats of labels
driver_act = read_data("driver_act.csv")
driver_act_stats = np.unique(driver_act["label"],return_counts=True)
# (array([0, 1, 2]), array([1649848,   86024,   15801])) (104:5.5:1)

driver_info = read_data("driver_info.csv")
distance_info = read_data("distance_info.csv")
driver_gps = read_data("driver_gps.csv")
owner_info = read_data("owner_info.csv")
owner_log = read_data("owner_log.csv")

# select features
sel_feature = ['id', 'start_city_id', 'end_city_id',
               'truck_length', 'truck_weight', 'cargo_capacipy', 'cargo_type',
               'expect_freight', 'truck_type_list', 'truck_type',
               'mileage', 'highway', 'lon', 'lat', 'freight_unit']
owner_log_compr = owner_log[sel_feature]

merged_info = pd.merge(driver_act, owner_log_compr, how='inner')
merged_info_sub = merged_info[["start_city_id","end_city_id"]]
start_list = set(np.unique(distance_info["start_city_id"]))

def get_real_mileage(i):
    """
    get REAL mileage from distance list
    """
    
    place_0 = merged_info_sub.iloc[i][0]
    place_1 = merged_info_sub.iloc[i][1]
    if(place_0 in start_list):
        distance_info_sub = distance_info[distance_info["start_city_id"] == place_0]
        mileage_sub = distance_info_sub[distance_info_sub["end_city_id"] == place_1]["distance"]
        
    else:
        distance_info_sub = distance_info[distance_info["end_city_id"] == place_0]
        mileage_sub = distance_info_sub[distance_info_sub["start_city_id"] == place_1]["distance"]
    
    return(mileage_sub)

def multicore(func):
    result = []
    pool = mp.Pool(processes=6)
    for y in tqdm(pool.imap(func, np.arange(len(merged_info)))):
        try:
            result.append(float(y))
        except TypeError:
            result.append(0)
        
    return(result)

real_mileage_list = multicore(get_real_mileage)

# now we add real mileage
merged_info["real_mileage"] = real_mileage_list
del merged_info["mileage"] # and delete the old one

# select features from driver info
driver_sel_features = ['user_id','truck_len']
driver_info_compr = driver_info[driver_sel_features]
merged_info = pd.merge(merged_info, driver_info_compr, how='inner')

# add new features
# difference of cargo length and truck length
len_diff = merged_info["truck_len"] - merged_info["truck_length"]
len_diff[len_diff < 0] = 0
merged_info["len_diff"] = len_diff
# cargo weight per truck length
# (truck_weight) / truck_length
merged_info["truck_weight"][merged_info["truck_weight"] > 200] = merged_info["truck_weight"][merged_info["truck_weight"] > 200] / 10000
wei_len = merged_info["truck_weight"] / (merged_info["truck_len"] + 0.1)
merged_info["wei_len"] = wei_len

# save merged files to csv
merged_info.to_csv("merged_info.csv")
merged_info = pd.read_csv("merged_info.csv")
del merged_info["Unnamed: 0"]

# downsampling for the training set (104:5.5:1)
merged_info_0 = merged_info[merged_info["label"] == 0]
merged_info_1 = merged_info[merged_info["label"] == 1]
merged_info_2 = merged_info[merged_info["label"] == 2]
# down sampling
merged_info_0_dn = merged_info_0.sample(frac=0.01, replace=False)
merged_info_1_dn = merged_info_1.sample(frac=0.18, replace=False)

# now we combine all down_sampled data
merged_info_dn = merged_info_2.append(merged_info_0_dn)
merged_info_dn = merged_info_dn.append(merged_info_1_dn)

# ... and separete labels
labels = merged_info_dn["label"]
train_data = merged_info_dn.drop(["user_id","id","start_city_id",
                                  "end_city_id","label","lon","lat",
                                  "truck_type_list","truck_type","freight_unit",
                                  "highway"], axis = 1)

# use cross-validation and use cross-validation
# here, we use 5-fold cv
param_grid = {
    "n_estimators": np.arange(2,80,10),
    'max_features': np.arange(1,9,2)
}

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
np.warnings.filterwarnings('ignore')
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_data, labels)
print(CV_rfc.best_params_)
# best params: max_features= 7,n_estimators=72

kf = KFold(n_splits=5,shuffle=True)
rmse_list = []
for train_index, test_index in kf.split(train_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    rfc = RandomForestClassifier(n_jobs=-1,max_features= 7,n_estimators=72, oob_score = True)
    rfc.fit(X_train, y_train)
    result = rfc.predict(X_test)
    result2 = rfc.predict_proba(X_test)
    
    ress = y_test - result
    ress[ress != 0] = 1
    
    rmse = np.sum(ress) / len(y_test)
    rmse_list.append(rmse)

rmse_avg = np.mean(rmse_list)
print(rmse_avg)
