import json, sys
import numpy as np
import pandas as pd
from statistics import mean, stdev, variance
import itertools as it
from os.path import join, exists
from sklearn.svm import SVC #, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import random

from utils import get_fold_ids


        
def MLalgos(
    model_name, folder_name,
    scaling, normalization, no_warnings=True
):
    if no_warnings:
        import warnings
        warnings.filterwarnings('ignore')
    
    data_path = f'./data/{folder_name}'
    n_feat = 1378
    num_repeats = 3
    seeds = random.sample(range(1, 100), 20)[:num_repeats]

    for cway in ['AD_SZ_HC', 'AD_SZ', 'AD_HC', 'AD_HCinBSNIP', 'SZ_HC', 'SZ_HCinADNI']:
        perf_logs = {}
        logs_path = f'logs/ML/{cway}__{folder_name}__{scaling}{normalization}__testlogs.json'
        
        accs = []
        specificities = []
        sensitivities = []
        for repeat_num, seed in enumerate(seeds):
            for fold_num in range(10):
                
                test_data, valid_data, train_data = get_fold_ids(
                    fold_num=fold_num, num_folds=10, seed=seed, cway=cway,
                )

                X_train = np.zeros(shape=(train_data.shape[0],n_feat))
                X_test = np.zeros(shape=(test_data.shape[0],n_feat))

                y_train = list(train_data.label.values)
                y_test = list(test_data.label.values)

                for ind, (_, row) in enumerate(train_data.iterrows()):
                    sFNC_dir = join(data_path, row.subject_id+'.npy')
                    sFNC = np.load(sFNC_dir).astype('float32')
                    # to extract the upper triangle values that are above the diagonal (k=1) to a flat vector
                    sFNC = sFNC[np.triu_indices(53, k = 1)] 
                    X_train[ind,:] = sFNC

                for ind, (_, row) in enumerate(test_data.iterrows()):
                    sFNC_dir = join(data_path, row.subject_id+'.npy')
                    sFNC = np.load(sFNC_dir).astype('float32')
                    # to extract the upper triangle values that are above the diagonal (k=1) to a flat vector
                    sFNC = sFNC[np.triu_indices(53, k = 1)]
                    X_test[ind,:] = sFNC

                if scaling:
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)

                # min-max normalization
                def normalize(data):
                    data_min = data.min(axis=1,keepdims=True)
                    data_max = data.max(axis=1,keepdims=True)
                    normalized_data = (data-data_min)/(data_max-data_min)
                    return normalized_data

                if normalization:
                    X_train = normalize(X_train)
                    X_test = normalize(X_test)
                
                logs = json.load(open(f'logs/ML/{cway}__{model_name}__{folder_name}__r{repeat_num}__{scaling}{normalization}__validlogs.json'))
                param = logs[f'params_f{fold_num}_r{repeat_num}']
                print(param)

                # apply ML model to training data
                if 'SVM' in model_name:
                    model = SVC( **param, random_state=0)
                elif model_name == 'LR':
                    model = LogisticRegression(**param, max_iter=1000, random_state=0)
                elif model_name == 'DT':
                    model = tree.DecisionTreeClassifier(**param, random_state=0)
                elif model_name == 'RF':
                    model = RandomForestClassifier(**param, random_state=0)
                elif model_name == 'KNN':
                    model = KNeighborsClassifier(**param)

                clf = model.fit(X_train,y_train)

                # predict using model
                y_pred = model.predict(X_test)
                accs.append(accuracy_score(y_test, y_pred)*100)

                if cway != 'AD_SZ_HC':
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    specificities.append((tn / (tn+fp))*100)
                    sensitivities.append((tp / (tp+fn))*100)

        if exists(logs_path):
            with open(logs_path, 'r+') as f:
                perf_logs = json.load(f)
        perf_logs.update({
            f'{model_name}__accs': accs,
            f'{model_name}__mean_acc': mean(accs), 
            f'{model_name}__stdev_acc': stdev(accs), 
            f'{model_name}__specificities': specificities, 
            f'{model_name}__sensitivities': sensitivities, 
        })
        with open(logs_path, 'w') as fp:
            json.dump(perf_logs, fp, indent=4, sort_keys=False) 


if __name__ == '__main__': 
    
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    
    model_name = sys.argv[1]
    folder_name = sys.argv[2]
    scaling = str2bool(sys.argv[3])
    normalization = str2bool(sys.argv[4])
    no_warnings = True
    
        
    MLalgos(model_name, folder_name, scaling, normalization, no_warnings)
    
    