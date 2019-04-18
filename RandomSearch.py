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
    
def add_quantiles(seg_id,feat_name,X,xc,ws=""):
    X.loc[seg_id,feat_name + "_25_quantile"+ws] = xc.quantile(0.25)
    X.loc[seg_id,feat_name + "_50_quantile"+ws] = xc.quantile(0.50)
    X.loc[seg_id,feat_name + "_75_quantile"+ws] = xc.quantile(0.75)
    X.loc[seg_id,feat_name + "_1_quantile"+ws] = xc.quantile(0.01)
    X.loc[seg_id,feat_name + "_99_quantile"+ws] = xc.quantile(0.99)
    X.loc[seg_id,feat_name + "_5_quantile"+ws] = xc.quantile(0.05)
    X.loc[seg_id,feat_name + "_95_quantile"+ws] = xc.quantile(0.95)
    
def create_features(seg_id,seg, X):
    xc = seg["acoustic_data"]
    
    fftxc = fft(xc)
    X.loc[seg_id,"fft_mean"] = np.abs(fftxc).mean()
    X.loc[seg_id,"fft_var"] = np.abs(fftxc).var()
    X.loc[seg_id,"fft_min"] = np.abs(fftxc).min()
    X.loc[seg_id,"fft_max"] = np.abs(fftxc).max()
    
    xc_splitted = np.array_split(xc, 4)
   
    add_statistics(seg_id,"1_part",X,xc_splitted[0])
    add_statistics(seg_id,"2_part",X,xc_splitted[1])
    add_statistics(seg_id,"3_part",X,xc_splitted[2])
    add_statistics(seg_id,"4_part",X,xc_splitted[3])
    
    add_quantiles(seg_id,"1_part",X,xc_splitted[0])
    add_quantiles(seg_id,"2_part",X,xc_splitted[1])
    add_quantiles(seg_id,"3_part",X,xc_splitted[2])
    add_quantiles(seg_id,"4_part",X,xc_splitted[3])
    
    fft_p1 = fft(xc_splitted[0])
    X.loc[seg_id,"fft_mean_p1"] = np.abs(fft_p1).mean()
    X.loc[seg_id,"fft_var_p1"] = np.abs(fft_p1).var()
    X.loc[seg_id,"fft_min_p1"] = np.abs(fft_p1).min()
    X.loc[seg_id,"fft_max_p1"] = np.abs(fft_p1).max()
    
    fft_p2 = fft(xc_splitted[1])
    X.loc[seg_id,"fft_mean_p2"] = np.abs(fft_p2).mean()
    X.loc[seg_id,"fft_var_p2"] = np.abs(fft_p2).var()
    X.loc[seg_id,"fft_min_p2"] = np.abs(fft_p2).min()
    X.loc[seg_id,"fft_max_p2"] = np.abs(fft_p2).max()
    
    fft_p3 = fft(xc_splitted[2])
    X.loc[seg_id,"fft_mean_p3"] = np.abs(fft_p3).mean()
    X.loc[seg_id,"fft_var_p3"] = np.abs(fft_p3).var()
    X.loc[seg_id,"fft_min_p3"] = np.abs(fft_p3).min()
    X.loc[seg_id,"fft_max_p3"] = np.abs(fft_p3).max()
    
    fft_p4 = fft(xc_splitted[3])
    X.loc[seg_id,"fft_mean_p4"] = np.abs(fft_p4).mean()
    X.loc[seg_id,"fft_var_p4"] = np.abs(fft_p4).var()
    X.loc[seg_id,"fft_min_p4"] = np.abs(fft_p4).min()
    X.loc[seg_id,"fft_max_p4"] = np.abs(fft_p4).max()
    
    add_statistics(seg_id,"seg",X,xc)
    add_quantiles(seg_id,"seg",X,xc)
    
    window_sizes = [20,200,1000]
    for window_size in window_sizes:
        xc_rolled = xc.rolling(window_size)
        xc_rolled_mean = xc_rolled.mean().dropna()
        xc_rolled_var = xc_rolled.var().dropna()
        xc_rolled_kurt = xc_rolled.kurt().dropna()
        xc_rolled_skew = xc_rolled.skew().dropna()
        ws = "_"+str(window_size)
        
        add_statistics(seg_id,"rollingMean",X,xc_rolled_mean,ws=ws)
        add_statistics(seg_id,"rollingVar",X,xc_rolled_var,ws=ws)
        add_statistics(seg_id,"rollingKurt",X,xc_rolled_skew,ws=ws)
        add_statistics(seg_id,"rollingSkew",X,xc_rolled_kurt,ws=ws)
        
        add_quantiles(seg_id,"rollingMean",X,xc_rolled_mean,ws=ws)
        add_quantiles(seg_id,"rollingVar",X,xc_rolled_var,ws=ws)
        add_quantiles(seg_id,"rollingKurt",X,xc_rolled_skew,ws=ws)
        add_quantiles(seg_id,"rollingSkew",X,xc_rolled_kurt,ws=ws)

for seg_id in range(segments):
    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]
    create_features(seg_id, seg,train_X)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    if seg_id % 500 == 0:
        print("iteration {}".format(seg_id))


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
        print("iteration {}/{}".format(it,nr_iterations))
        params = {
            'boosting_type': rand.choice(['gbdt','rf']),
            'objective': 'regression',
            'min_data_in_leaf': randint(1,100), 
            'num_leaves': randint(2,100),
            'max_depth': rand.choice([-1,randint(2,20)]),
            'learning_rate': rand.uniform(0.0005,0.007),
            'feature_fraction': rand.uniform(0.5,0.98),
            'bagging_fraction': rand.uniform(0.5,0.98),
            'bagging_freq': randint(1,20),
            'lambda_l1':rand.uniform(0,0.01),
            'lambda_l2':rand.uniform(0,0.01),
            'metric':'mae',
            'verbose':-1
        }
        train_score = cross_validate(train_X,train_y,params)
        print("params = {}".format(params))
        if train_score < best_score:
            print("updated score from {} to {}".format(best_score,train_score))
            best_params = params
            best_score = train_score
        else:
            print("score = {}, leaving {}".format(train_score,best_score))
         

    print("Best score achieved {} for params\n {}".format(best_score,best_params))

random_search(train_X,train_y,2000)