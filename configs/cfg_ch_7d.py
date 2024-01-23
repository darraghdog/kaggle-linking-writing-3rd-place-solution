import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import torch
import pandas as pd
import glob
sys.path.append("configs")
sys.path.append("data")
sys.path.append("models")
sys.path.append("scripts")

#from default_nlp_config import basic_cfg#, basic_nlp_cfg
from transformers.models.speech_to_text import Speech2TextConfig
from types import SimpleNamespace
from transformers import Wav2Vec2Processor, Wav2Vec2Config, AutoTokenizer
import augmentations as A

cfg = SimpleNamespace(**{})
cfg.debug = True

BASEPATH = './'
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"/raid1/writing_quality/weights/{os.path.basename(__file__).split('.')[0]}"
cfg.data_folder = f"{BASEPATH}/datamount/"
cfg.train_df = f'{BASEPATH}/datamount/train_v03.csv.gz'

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_folder
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1


#logging
cfg.neptune_project = "light/writingquality"
cfg.neptune_connection_mode = "async"
cfg.tags = "train"

cfg.dataset = "ds_dh_08b" #"ds_dh_2j"

#model
cfg.model = "mdl_ch_5b"
#cfg.backbone = 'microsoft/deberta-v3-base'
#cfg.backbone = 'weights/pretrain_mlm_v01/checkpoint-1000/'
cfg.backbone = 'weights/pretrain_mlm_v06e4/microsoft/deberta-v3-base/checkpoint-3060/'
cfg.backbone_cfg = {"attention_probs_dropout_prob": 0.,
                    "hidden_dropout_prob": 0.}
cfg.feat_dim = 256
cfg.feat_init_ksize = 21

# DATASET
# cfg.classes = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
cfg.n_classes = 6
cfg.label_col = 'score'
# cfg.data_sample = 1000
cfg.tokenizer = 'microsoft/deberta-v3-base'
cfg.padding = True
cfg.truncation = True
cfg.max_length = 3200
cfg.group_by_length = False
cfg.replace_char = 'X'
cfg.break_char = '|'
cfg.round_cumtimes = 500
cfg.round_uptimes = 1000
cfg.max_times = 3600

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 7

cfg.lr = 2e-5
cfg.optimizer = "Adam"
cfg.weight_decay = 1e-3
cfg.warmup = 1.0
cfg.batch_size = 4
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 4.
cfg.num_workers = 8
cfg.gradient_checkpointing = True

# cfg.dropout_mha = 0.2
cfg.dropout_mha2 = torch.nn.Dropout(0.3)

# EVAL

cfg.post_process_pipeline =  "pp_dh_01"
# cfg.dummy_phrase_ids = [char_to_num[c] for c in '2 a-e -aroe']
cfg.metric = "default_metric"
# augs & tta
cfg.calc_metric = True
cfg.simple_eval = False
# cfg.calc_metric_epochs = 30 
# augs & tta

# Postprocess

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.train_aug = None
cfg.val_aug = None
cfg.train_aug = A.Compose([A.TemporalMask2(size=(0.2,0.4),num_masks=(1,4),mask_value=0.,p=0.6), #mask with 0 as it is post-normalization
                          ])
cfg.train_aug._disable_check_args() #disable otherwise input must be numpy/ int8
cfg.val_aug = None
