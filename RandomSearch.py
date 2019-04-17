import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.tree import export_graphviz
import lightgbm as lgb
from scipy.fftpack import fft

from random import randint
import random as rand


import gc

print("Reading data...")
train_df = pd.read_csv(os.path.join("../input",'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)

train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

def add_statistics(seg_id,feat_name,X,xc,ws=""):
    X.loc[seg_id,feat_name + "_mean"+ws] = xc.mean()
    X.loc[seg_id,feat_name + "_var"+ws] = xc.var()
    X.loc[seg_id,feat_name + "_min"+ws] = xc.min()
    X.loc[seg_id,feat_name + "_max"+ws] = xc.max()
    X.loc[seg_id,feat_name + "_kurt"+ws] = xc.kurt()
    X.loc[seg_id,feat_name + "_skew"+ws] = xc.skew()
    
def create_features(seg_id,seg, X):
    xc = seg["acoustic_data"]
    
    fftxc = fft(xc)
    X.loc[seg_id,"fft_mean"] = np.abs(fftxc).mean()
    X.loc[seg_id,"fft_max"] = np.abs(fftxc).max()
    X.loc[seg_id,"fft_min"] = np.abs(fftxc).min()
    
    X.loc[seg_id,"25_quantile"] = xc.quantile(0.25)
    X.loc[seg_id,"50_quantile"] = xc.quantile(0.50)
    X.loc[seg_id,"75_quantile"] = xc.quantile(0.75)
    X.loc[seg_id,"1_quantile"] = xc.quantile(0.01)
    X.loc[seg_id,"99_quantile"] = xc.quantile(0.99)
    
    xc_splitted = np.array_split(xc, 4)
    X.loc[seg_id,"1_part_mean"] = xc_splitted[0].mean()
    X.loc[seg_id,"2_part_mean"] = xc_splitted[1].mean()
    X.loc[seg_id,"3_part_mean"] = xc_splitted[2].mean()
    X.loc[seg_id,"4_part_mean"] = xc_splitted[3].mean()
    X.loc[seg_id,"1_part_var"] = xc_splitted[0].var()
    X.loc[seg_id,"2_part_var"] = xc_splitted[1].var()
    X.loc[seg_id,"3_part_var"] = xc_splitted[2].var()
    X.loc[seg_id,"4_part_var"] = xc_splitted[3].var()
    
    #add_statistics(seg_id,"1_part",X,xc_splitted[0])
    #add_statistics(seg_id,"2_part",X,xc_splitted[1])
    #add_statistics(seg_id,"3_part",X,xc_splitted[2])
    #add_statistics(seg_id,"4_part",X,xc_splitted[3])
    X.loc[seg_id,"avg"] = xc.mean()
    X.loc[seg_id,"max"] = xc.max()
    X.loc[seg_id,"min"] = xc.min()
    X.loc[seg_id,"skew"] = xc.skew()
    X.loc[seg_id,"kurt"] = xc.kurt()
    X.loc[seg_id,"var"] = xc.var()
    
    #add_statistics(seg_id,"seg",X,xc)
    
    #window_sizes = [20,100,500,1000]
    #for window_size in window_sizes:
    #    xc_rolled = xc.rolling(window_size)
    #    xc_rolled_mean = xc_rolled.mean().dropna()
    #    xc_rolled_var = xc_rolled.var().dropna()
    #    ws = "_"+str(window_size)
    #    
    #    add_statistics(seg_id,"rollingMean",X,xc_rolled_mean,ws=ws)
    #    add_statistics(seg_id,"rollingVar",X,xc_rolled_var,ws=ws)
        
        #X.loc[seg_id,"rolling_mean_"+ws] = xc_rolled_mean.mean()
        #X.loc[seg_id,"rolling_max_"+ ws] = xc_rolled_mean.max()
        #X.loc[seg_id,"rolling_min_"+ws] = xc_rolled_mean.min()
        #X.loc[seg_id,"rolling_skew_"+ws] = xc_rolled_mean.skew()
        #X.loc[seg_id,"rolling_kurt_"+ws] = xc_rolled_mean.kurt()
        #X.loc[seg_id,"rolling_var_"+ws] = xc_rolled_mean.var()
        #
        #X.loc[seg_id,"rollingvar_mean_"+ws] = xc_rolled_var.mean()
        #X.loc[seg_id,"rollingvar_max_"+ ws] = xc_rolled_var.max()
        #X.loc[seg_id,"rollingvar_min_"+ws] = xc_rolled_var.min()
        #X.loc[seg_id,"rollingvar_skew_"+ws] = xc_rolled_var.skew()
        #X.loc[seg_id,"rollingvar_kurt_"+ws] = xc_rolled_var.kurt()
        #X.loc[seg_id,"rollingvar_var_"+ws] = xc_rolled_var.var()
        
for seg_id in tqdm(range(segments)):
    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
    create_features(seg_id, seg,train_X)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    
def load_test():
    signals = []
    segment_names = [file for file in os.listdir("../input") if file.startswith("seg")]
    test_df = pd.DataFrame(index=segment_names, dtype=np.float64)
    test_df.index = test_df.index.str[:-4]
    for file in tqdm(segment_names):
        seg_id = file[:-4]
        segment = pd.read_csv(os.path.join("../input",file),dtype={'acoustic_data': np.int16})
        create_features(seg_id,segment,test_df)
    return test_df
test_df = load_test()


def cross_validate(train_X,train_y,params):
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    oof = np.zeros(train_X.shape[0])
    for train_idx,val_idx in folds.split(train_X,train_y):
        X_train,y_train = train_X.iloc[train_idx],train_y.iloc[train_idx]
        X_val,y_val = train_X.iloc[val_idx],train_y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val,y_val, reference=lgb_train)
        model = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)
        model.fit(X_train,y_train,
                  eval_set=[(X_train,y_train),(X_val,y_val)], 
                  verbose=1000,
                  early_stopping_rounds=500)

        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    return mean_absolute_error(train_y,oof)

def random_search(train_X,train_y,nr_iterations):
    best_score = np.Infinity
    best_params = None
    for it in range(nr_iterations):
        params = {
            'boosting_type': rand.choice(['gbdt','rf']),
            'objective': 'regression',
            'min_data_in_leaf': randint(1,100), 
            'num_leaves': randint(1,100),
            'max_depth': rand.choice([-1,randint(1,20)]),
            'learning_rate': rand.uniform(0.0005,0.007),
            'feature_fraction': rand.uniform(0.5,0.98),
            'bagging_fraction': rand.uniform(0.5,0.98),
            'bagging_freq': randint(1,20),
            'verbose': 1,
            'lambda_l1':rand.uniform(0,0.01),
            'lambda_l2':rand.uniform(0,0.01),
            'metric':'mae'
        }
        train_score = cross_validate(train_X,train_y,params)
        if train_score < best_score:
            best_params = params
            best_score = train_score
            print("updated score from {} to {}".format(best_score,train_score))
        else:
            print("score = {}, leaving {}".format(train_score,best_score))
         

    print("Best score achieved {} for params\n {}".format(best_score,best_params))

random_search(train_X,train_y,100)