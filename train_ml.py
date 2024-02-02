import json, sys, os
import numpy as np
import pandas as pd
from statistics import mean, stdev, variance
import itertools as it
from os.path import join, exists
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import random

from utils import get_fold_ids


        
def MLalgos(
    model_name, cway, folder_name,
    repeat_num, seed,
    scaling, normalization, no_warnings=True
):

    if no_warnings:
        import warnings
        warnings.filterwarnings('ignore')
    
    perf_logs = {}
    logs_path = f'logs/ML/{cway}__{model_name}__{folder_name}__r{repeat_num}__{scaling}{normalization}__validlogs.json'
    if not exists('logs/ML'):
        os.makedirs('logs/ML')
    data_path = f'./data/{folder_name}'
    n_feat = 1378
            
    for fold_num in range(10):
        
        # hyper params
        if model_name == 'SVM':
            C_range = np.logspace(-4, 2, 7)
            gamma_range = ['scale', 'auto']
            coef0_range = [0.0, 0.1]
            degree_range = range(10)  

            param_grid = {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed'] 
                'C': C_range, 'gamma' : gamma_range, 
                'degree' : degree_range, 'coef0': coef0_range,
            }
            
        elif model_name == 'LR':
            C_range = 10. ** np.arange(-3, 3)
            l1_ratio = np.linspace(0, 1, 5)
            
            param_grid = {
                'solver': ['newton-cg', 'lbfgs', 'sag'], 
                'C': C_range, 'penalty': ['none', 'l2'],
            }

        elif model_name == 'DT':
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [1, 2, 3, 4, 5, 10, 20, 30, None], 
                'min_samples_split': [2, 3, 4, 5, 10], 
                'min_samples_leaf': [1, 2, 3, 4, 5],
            }

        elif model_name == 'RF':
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'n_estimators': [100, 200, 300, 400, 500],
    #             'max_features': ['auto', 'sqrt'],
                'max_depth': [1, 3, 5, 10, 20, 30, 40, 50, 100, None], 
                'min_samples_split': [2, 5, 10], 
                'min_samples_leaf': [1, 2, 4],
            }
        elif model_name == 'KNN':
            param_grid = {
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
                'weights': ['uniform', 'distance'],
                'leaf_size': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
                'p': [1,2],
            }
        
        allNames = param_grid.keys()
        combinations = it.product(*(param_grid[Name] for Name in allNames))

        best_accs = 0
        for param in combinations: 
            param_grid.update(zip(param_grid.keys(),param))

            test_data, valid_data, train_data = get_fold_ids(
                fold_num=fold_num, num_folds=10, seed=seed, cway=cway,
            )

            X_train = np.zeros(shape=(train_data.shape[0],n_feat))
            X_valid = np.zeros(shape=(valid_data.shape[0],n_feat))

            y_train = list(train_data.label.values)
            y_valid = list(valid_data.label.values)

            for ind, (_, row) in enumerate(train_data.iterrows()):
                sFNC_dir = join(data_path, row.subject_id+'.npy')
                sFNC = np.load(sFNC_dir).astype('float32')
                # to extract the upper triangle values that are above the diagonal (k=1) to a flat vector
                sFNC = sFNC[np.triu_indices(53, k = 1)] 
                X_train[ind,:] = sFNC

            for ind, (_, row) in enumerate(valid_data.iterrows()):
                sFNC_dir = join(data_path, row.subject_id+'.npy')
                sFNC = np.load(sFNC_dir).astype('float32')
                # to extract the upper triangle values that are above the diagonal (k=1) to a flat vector
                sFNC = sFNC[np.triu_indices(53, k = 1)] 
                X_valid[ind,:] = sFNC

            if scaling:
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_valid = sc.transform(X_valid)

            # min-max normalization
            def normalize(data):
                data_min = data.min(axis=1,keepdims=True)
                data_max = data.max(axis=1,keepdims=True)
                normalized_data = (data-data_min)/(data_max-data_min)
                return normalized_data

            if normalization:
                X_train = normalize(X_train)
                X_valid = normalize(X_valid)

            # apply ML model to training data
            if 'SVM' in model_name:
                model = SVC(**param_grid, random_state=0)
            elif model_name == 'LR':
                model = LogisticRegression(**param_grid, max_iter=1000, random_state=0)
            elif model_name == 'DT':
                model = tree.DecisionTreeClassifier(**param_grid, random_state=0)
            elif model_name == 'RF':
                model = RandomForestClassifier(**param_grid, random_state=0)
            elif model_name == 'KNN':
                model = KNeighborsClassifier(**param_grid)

            clf = model.fit(X_train,y_train)

            # predict using model
            y_pred = model.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)*100

            if best_accs < acc:
                best_accs = acc
                best_param = param_grid.copy()

        if exists(logs_path):
            with open(logs_path, 'r+') as f:
                perf_logs = json.load(f)
        perf_logs.update({
            f'acc_f{fold_num}_r{repeat_num}': best_accs, 
            f'params_f{fold_num}_r{repeat_num}': best_param,
            f'model_name_f{fold_num}_r{repeat_num}': model_name,
        })
        with open(logs_path, 'w') as fp:
            json.dump(perf_logs, fp, indent=4, sort_keys=False) 


if __name__ == '__main__': 
    
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    
    model = sys.argv[1]
    cway = sys.argv[2]
    folder_name = sys.argv[3]
    repeat_num = int(sys.argv[4])
    seed = int(sys.argv[5])
    scaling = str2bool(sys.argv[6])
    normalization = str2bool(sys.argv[7])
    no_warnings = True
            
    MLalgos(model, cway, folder_name, repeat_num, seed, scaling, normalization, no_warnings)
    
    