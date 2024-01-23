import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

'''
val_df =  pd.read_csv(cfg.train_df).query('fold==0').set_index('id')
val_data = torch.load('weights/cfg_dh_01/fold0/val_data_seed130970.pth', map_location=torch.device('cpu'))

scores = val_df.drop_duplicates('idx')['score'].values
dev = torch.device('cpu')
fnms = glob.glob('weights/cfg_dh_05b7/fold0/val*') +\
    glob.glob('weights/cfg_dh_05b3/fold0/val*') +\
        glob.glob('weights/cfg_dh_05b5/fold0/val*')
    
pred = torch.stack([torch.load(fnm, map_location=dev)['logits'].flatten() \
             for fnm in fnms])
pred1 = pred.mean(0).float().numpy()
np.sqrt(np.mean(((pred1-scores)**2)))


pred2 = torch.pow(torch.prod(pred.float(), dim=0), 1/len(pred)).flatten().numpy()

np.sqrt(np.mean(((pred2-scores)**2)))




folddf =  pd.read_csv(cfg.train_df, usecols = ['id', "fold", "score"]).drop_duplicates().reset_index(drop = True)
pred2 = pd.read_csv("~/Downloads/xgb_cv.csv").xgb.values
np.sqrt(np.mean(((folddf.score-pred2 )**2)))

pred2a = pred2[(folddf.fold==0)]
'''
def post_process_pipeline(cfg, val_data, val_df):
    
    print('postprocessing...')
    
     
    ids = val_df.index.unique()
    scores = val_df.drop_duplicates('idx')['score'].values
    pred = val_data['logits'].detach().cpu().numpy().flatten()
    
    pp_out = pd.DataFrame({'ids': ids, 'act': scores, 'prediction': pred})
    # outdf.plot.scatter('act', 'prediction')
    
    # np.sqrt(np.mean(((pred-scores)**2)))
    # np.sqrt(np.mean(((pred2a-scores)**2)))
    
    # np.sqrt(np.mean((((pred + pred2a)/2-scores)**2)))
    
    return pp_out