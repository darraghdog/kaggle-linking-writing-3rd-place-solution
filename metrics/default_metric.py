import numpy as np
from sklearn.metrics import f1_score
import torch
import scipy as sp
import pandas as pd
import os, sys, importlib, copy, torch, importlib, glob


'''
def get_score(y_true, y_pred):
    return 0
#     score = sp.stats.pearsonr(y_true, y_pred)[0]
#     return score

def get_score(solution, submission):
    joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}), \
                            left_index=True, right_index=True)
    domain_scores = joined.groupby('domain').apply(
        # note that jiwer.wer computes a weighted average wer by default when given lists of strings
        lambda df: jiwer.wer(df['sentence'].to_list(), df['predicted'].to_list()),
    )
    return domain_scores.mean()

'''

def calc_metric(cfg, pp_out, val_df, pre="val"):
    
    score = np.sqrt(np.mean((pp_out['act'] - pp_out['prediction'])**2))
    
    return score



