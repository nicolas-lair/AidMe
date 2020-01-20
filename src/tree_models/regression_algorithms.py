#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib


def save_model(model, path):
    """Save a trained model

    Args:
        path(str): Path to where a learnt model is stored
    """
    result = joblib.dump(model, path)
    return result


def load_model(path):
    """Load a trained model

    Args:
        path(str): Path to where a learnt model is stored
    
    Returns:
        object: Saved model
    """
    trained_model = joblib.load(path)
    return trained_model


def train_randomforest(train_features, train_labels, max_depth=8, n_estimators=20):
    """Train random forest model with default settings

    Args:
        train_features(np.ndarray): n*m matrix with n the number of samples
        train_labels(np.ndarray): n labels associated to the features
    
    Returns:
        model: Random Forest trained model
    """
    # Random Forest expect a 1D array as input for labels, hence this 
    # Simple check
    train_features = np.nan_to_num(train_features)
    if len(train_labels.shape) == 2 and train_labels.shape[1] == 1:
        train_labels = train_labels.reshape(train_labels.shape[0], )
    rgs = RandomForestRegressor(n_estimators=n_estimators, criterion='mse', max_depth=max_depth, random_state=0)
    rgs.fit(train_features, train_labels)
    return rgs


def train_xgboost_scikit(train_features, train_labels, max_depth=8, **kwargs):
    """Train random forest model with default settings::
        {
            'max_depth': 3, 
            'eta': 0.1, 
            'silent': 1, 
            'objective': 'reg:linear',
            'nthread': 4,
            'eval_metric': 'rmse'
        }

    Args:
        train_features(np.ndarray): n*m matrix with n the number of samples
        train_labels(np.ndarray): n labels associated to the features
    
    Returns:
        model: XGBoost trained model
    """
    xgr = xgb.XGBRegressor(max_depth=max_depth, learning_rate=0.1, silent=True, objective='reg:squarederror', nthread=4)
    xgr.fit(train_features, train_labels)
    return xgr


def train_xgboost(train_features, train_labels, dev_features=None, dev_labels=None):
    """Train random forest model with default settings::
        {
            'max_depth': 3, 
            'eta': 0.1, 
            'silent': 1, 
            'objective': 'reg:linear',
            'nthread': 4,
            'eval_metric': 'rmse'
        }
    
    The default is a XGBoost model, therefore only DMatrix can be used with it once it is 
    fitted, not numpy arrays as for scikitlearn models.
    Args:
        train_features(np.ndarray): n*m matrix with n the number of samples
        train_labels(np.ndarray): n labels associated to the features
    
    Returns:
        model: XGBoost trained model
    """
    dtrain = xgb.DMatrix(train_features, train_labels)
    param = {
        'max_depth': 3,
        'eta': 0.1,
        'silent': 1,
        'objective': 'reg:linear',
        'nthread': 4,
        'eval_metric': 'rmse'
    }
    num_round = 1000
    if dev_features is not None and dev_labels is not None:
        ddev = xgb.DMatrix(dev_features, dev_labels)
        evallist = [(dtrain, 'train'), (ddev, 'eval')]
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    else:
        bst = xgb.train(param, dtrain, num_round)
    return bst
