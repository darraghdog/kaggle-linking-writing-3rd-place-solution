import numpy as np
import re
import os
import torch
from torch.utils.data import Dataset
from transformers.models.auto import AutoTokenizer
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def batch_to_device(batch, device):
    

    batch_dict = {key: batch[key].to(device) for key in ['input_ids','attention_mask', "uptimes", "cumtimes", "cursor", 'idx','target']}
    return batch_dict

from torch.nn.functional import pad

def collate_fn(batch):
    l = 0
    for b in batch:
        this_ids = b["input_ids"].shape[0]
        if this_ids > l:
            l = this_ids
    out_dict = {
        "input_ids": torch.stack([pad(b["input_ids"],(0, l - b["input_ids"].shape[0]),mode="constant",value=1,) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"],(0, l - b["input_ids"].shape[0]),mode="constant",value=0,) for b in batch]),
        "uptimes": torch.stack([pad(b["uptimes"],(0, l - b["uptimes"].shape[0]),mode="constant",value=0,) for b in batch]),
        "cumtimes": torch.stack([pad(b["cumtimes"],(0, l - b["cumtimes"].shape[0]),mode="constant",value=0,) for b in batch]),
        "cursor": torch.stack([pad(b["cursor"],(0, l - b["cursor"].shape[0]),mode="constant",value=0,) for b in batch]),
        "idx": torch.stack([b["idx"] for b in batch]),
    }

    if "target" in batch[0].keys():
        
        out_dict.update(
            {
                "target": torch.stack([b["target"] for b in batch])
            }
        )

    return out_dict

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

def getEssays2(dd):
    
    # Copy required columns
    textInputDf = dd[['activity', 'cursor_position', 'text_change', 'up_time', 'action_time']].copy()
    currTextInput = textInputDf[textInputDf.activity != 'Nonproduction']
    currTextTimes = dd[['up_time', 'action_time']].copy()
    currTextTimes['action_time']
    textInputDf = dd[['activity', 'cursor_position', 'text_change']].copy()

    # Where the essay content will be stored
    essayText = ""
    upTimes2 = np.zeros((currTextInput.cursor_position.max()))
    cumTimes2 = np.zeros((currTextInput.cursor_position.max()))
    # Produces the essay
    curr_time = 0
    for Input, InputTime in zip(currTextInput.values, currTextTimes.values):
        # Input[0] = activity
        # Input[2] = cursor_position
        # Input[3] = text_change
        # If activity = Replace
        upTimes2[Input[1]-1] = InputTime[0]
        cumTimes2[Input[1]-1] += InputTime[0] - curr_time
        curr_time = InputTime[0]
        
        if Input[0] == 'Replace':
            # splits text_change at ' => '
            replaceTxt = Input[2].split(' => ')
            
            # DONT TOUCH
            from_txt, insert_ = essayText[:Input[1] - len(replaceTxt[1])], replaceTxt[1]
            essayText = from_txt + insert_ + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue

            
        # If activity = Paste    
        if Input[0] == 'Paste':
            # DONT TOUCH
            from_txt, insert_ = essayText[:Input[1] - len(Input[2])], Input[2]
            essayText = from_txt + insert_ + essayText[Input[1] - len(Input[2]):]
            continue

            
        # If activity = Remove/Cut
        if Input[0] == 'Remove/Cut':
            # DONT TOUCH
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue

            
        # If activity = Move...
        if "M" in Input[0]:
            # Gets rid of the "Move from to" text
            croppedTxt = Input[0][10:]
            
            # Splits cropped text by ' To '
            splitTxt = croppedTxt.split(' To ')
            
            # Splits split text again by ', ' for each item
            valueArr = [item.split(', ') for item in splitTxt]
            
            # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))

            # Skip if someone manages to activiate this by moving to same place
            if moveData[0] != moveData[2]:
                # Check if they move text forward in essay (they are different)
                if moveData[0] < moveData[2]:
                    # DONT TOUCH
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]

                else:
                    # DONT TOUCH
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                    
            continue
        
        # If just input
        # DONT TOUCH
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
        #upTimes[len(essayText[:Input[1] - len(Input[2])]): len(essayText[:Input[1] - len(Input[2])]) + len( Input[2])] += InputTime[1]
    
    upTimes2, cumTimes2 = upTimes2[:len(essayText)], cumTimes2[:len(essayText)]
    # Returns the essay series
    return essayText, upTimes2, cumTimes2


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.ids = self.df["id"].unique()
        self.df = self.df.set_index("id")
        self.aug = aug

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_length = cfg.max_length
        idx = 401

    def __getitem__(self, idx):
                
        # lenls = []
        rows = self.df.loc[self.ids[idx]]

        reconstructed, uptimes_calc, cumtimes_calc = getEssays2(rows.copy())
        reconstructed = reconstructed.replace('q', self.cfg.replace_char)
        reconstructed = reconstructed.replace('\n', self.cfg.break_char)
        
        uptimes, cumtimes = np.zeros(len(reconstructed)), np.zeros(len(reconstructed))
        uptimes[:len(uptimes_calc)], cumtimes[:len(cumtimes_calc)] = uptimes_calc, cumtimes_calc 
        
        feature_dict = self.tokenizer(reconstructed, 
                                      max_length=self.max_length,
                                      truncation=True,
                                      return_offsets_mapping = True,
                                      return_tensors = "pt")
        
        offsets = feature_dict['offset_mapping'][0]
        uptimes = uptimes[(offsets[:,0]-1).clip(0, 100000)]
        uptimes[0], uptimes[-1] = 0,0
        
        cumtimes = np.array([cumtimes[slice(*pos)].sum() for pos in offsets.numpy()])
        
        cumtimes = (cumtimes/self.cfg.round_cumtimes)#.round().astype(int)
        uptimes = (uptimes/self.cfg.round_uptimes)#.round().astype(int)
        uptimes = torch.from_numpy(uptimes)
        cumtimes = torch.from_numpy(cumtimes)
                
        
        # cursor
        cursor_orig = torch.from_numpy(rows.cursor_position.values).float()
        cursor = torch.zeros(len(offsets), dtype = torch.long)
        if (len(offsets) - 2)>0:
            cursor2 = torch.nn.functional.interpolate(cursor_orig[None,None,:], size=len(offsets) - 2, mode='linear')[0,0]
            cursor[1:len(offsets)-1] = cursor2#.round().long()
        #pd.Series(cursor.numpy()).plot()
        #pd.Series(cursor_orig.numpy()).plot()
        
        if self.aug:
            augfeatfn = lambda m: self.aug(image=m[None,:])['image'].flatten()
            uptimes = augfeatfn(uptimes)
            cumtimes = augfeatfn(cumtimes)
            cursor = augfeatfn(cursor)
        
        feature_dict = {k:v.flatten() for k,v in feature_dict.items()}
        feature_dict['uptimes'] = uptimes
        feature_dict['cumtimes'] = cumtimes
        feature_dict['cursor'] = cursor
        feature_dict['idx'] = torch.tensor(rows['idx'].values[0])
        feature_dict['target'] = torch.tensor(rows['score'].values[0]).float()
        if 'token_type_ids' in feature_dict: del feature_dict['token_type_ids']
        #,lenls.append(len(feature_dict['input_ids']))
        
        return feature_dict

    def __len__(self):
        return len(self.ids)

    def get_token_ids(self, text, max_length):

        out = self.tokenizer(
            text,
            return_tensors=None,
            return_offsets_mapping=False,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,  # leave 2 for special tokens
        )

        input_ids = out["input_ids"]
        if isinstance(input_ids, int):
            input_ids = [input_ids]
        
        return input_ids