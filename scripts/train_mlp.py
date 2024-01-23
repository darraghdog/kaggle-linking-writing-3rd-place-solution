import joblib
import pandas as pd
import numpy as np
import re
import torch
import glob
from tqdm import tqdm
from scripts.silver_bullet_feats_v01 import make_train_test_features

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

TIMEOUT = 300
N_THREADS = 32
N_FOLDS = 10
RANDOM_STATE = 23

oof_pred, automl = joblib.load('oof_and_lama_denselight_0.pkl')

'''
automl.general_params
automl.nn_params
automl.nn_pipeline_params
automl.reader_params
automl.task.name
automl.task.losses['torch'].loss
automl.cpu_limit
automl.timing_params
automl.tuning_params
automl.timer.timeout
'''

train_feats, test_feats = make_train_test_features(f"train_logs.csv", f"test_logs.csv")
trndf = pd.read_csv('datamount/train_v02.csv.gz')
folds = trndf[['id','fold']].drop_duplicates()
train_scores = trndf[['id','score']].drop_duplicates()
data = train_feats.merge(train_scores, on='id', how='left')
data = data.merge(folds, on='id', how='left')

# task = Task('reg')
roles = {
    'target': 'score',
    'drop': ['id','fold']
}

# mkdir weights/out_models

logits = []
oof_df = []

for fold in range(1+trndf.fold.max()):
    train = data[data['fold']!=fold].copy()
    val = data[data['fold']==fold].copy()

    x_tr  = train.drop(['id','fold','score'], axis=1)
    y_tr  = train['score'].values
    x_val = val.drop(['id','fold','score'], axis=1)
    y_val = val['score'].values
    
    model = TabularAutoML(
        task = automl.task, 
        timeout = TIMEOUT,
        tuning_params = automl.tuning_params,
        cpu_limit = automl.cpu_limit,
        general_params = automl.general_params, # ['nn', 'mlp', 'dense', 'denselight', 'resnet', 'snn', 'node', 'autoint', 'fttransformer'] or custom torch model
        nn_params = automl.nn_params,
        nn_pipeline_params = automl.nn_pipeline_params,
        reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE+fold}
    )
    
    _ = model.fit_predict(train, roles = roles, verbose = 1)
    joblib.dump(model, f'weights/denselight/ch_denselight_1_fold{fold}_seed{RANDOM_STATE+fold}.pkl')
    p = model.predict(x_val).data[:,0]
    
    logits +=[p] 
    oof_df += [val]
logits = np.concatenate(logits)
oof_df = pd.concat(oof_df)

print(f"RMSE: {rmse(oof_df['score'].values,logits)}")
oof_df['pred'] = logits
oof_df[['id','pred']].to_csv('oof.csv', index=False)
