#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
from random import randint
import random as rand

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np 

import lightgbm as lgb

from catboost import CatBoostRegressor


# In[83]:


train_X = pd.read_csv("../extracted_feat/train_58.csv")
train_y = pd.read_csv("../extracted_feat/train_y_58.csv")


# In[84]:


def get_random_cols(train_X):
    columns = train_X.columns.values
    test_columns = np.random.choice(columns,randint(3,columns.shape[0]-1),replace=False)
    return test_columns


# In[95]:


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
def train_cat(train_X,train_y):
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    oof = np.zeros(train_X.shape[0])
    params = {'loss_function':'MAE'}
    for train_idx,val_idx in folds.split(train_X,train_y):
        X_train,y_train = train_X.iloc[train_idx],train_y.iloc[train_idx]
        X_val,y_val = train_X.iloc[val_idx],train_y.iloc[val_idx]

        model = CatBoostRegressor(iterations=6000,  eval_metric='MAE', **params)
        model.fit(X_train,
                  y_train,
                  eval_set=(X_val, y_val),
                  cat_features=[], use_best_model=True, verbose=2000)

        oof[val_idx] = model.predict(X_val)

    return mean_absolute_error(train_y,oof)
def random_search(train_X,train_y,nr_iterations):
    best_score = np.Infinity
    best_cols = None
    for it in range(nr_iterations):
        print("iteration {}/{}".format(it,nr_iterations))
        params = {
            'lambda_l1': 0.012465994599126015, 
            'bagging_freq': 15, 
            'verbose': -1,
            'min_data_in_leaf': 5,
            'feature_fraction': 0.7143153769050614,
            'objective': 'MAE', 
            'lambda_l2': 0.055052283158846985, 
            'metric': 'MAE', 
            'bagging_fraction': 0.4871803105884792, 
            'max_depth': -1, 
            'learning_rate': 0.007017896834582354, 
            'boosting_type': 'gbdt',
            'num_leaves': 9
        #     'boost_from_average':False
        }
        
        random_cols = get_random_cols(train_X)
        train_score = train_cat(train_X[random_cols],train_y)
        print("cols = {}".format(random_cols))
        if train_score < best_score:
            print("updated score from {} to {}".format(best_score,train_score))
            best_cols = random_cols
            best_score = train_score
        else:
            print("score = {}, leaving {}".format(train_score,best_score))
         

    print("Best score achieved {} for cols\n {}".format(best_score,best_cols))


# In[ ]:


random_search(train_X,train_y,1000)


# In[ ]:




