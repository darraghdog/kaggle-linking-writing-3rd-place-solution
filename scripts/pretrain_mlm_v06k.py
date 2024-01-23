from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import warnings
import os
from transformers import AutoModelForMaskedLM, TrainingArguments
import argparse
from transformers import Trainer, DataCollatorForLanguageModeling, AutoTokenizer
import string
import pyarrow.parquet as pq
import gc
import glob
from tqdm import tqdm

warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, texts_pair=None, max_length=512):
        super().__init__()
        
        self.texts = texts
        self.texts_pair = texts_pair
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if self.texts_pair is not None:
            assert len(self.texts) == len(self.texts_pair)
        
    def __len__(self):
        return len(self.texts)
    
    def tokenize(self, text, text_pair=None):
        return self.tokenizer(
            text=text, 
            text_pair=text_pair,
            max_length=self.max_length,
            truncation=True,
            padding=False, 
            return_attention_mask=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_tensors=None,
        )
    
    def __getitem__(self, index):
        text = self.texts[index]
        
        text_pair = None
        if self.texts_pair is not None:
            text_pair = self.texts_pair[index]
            
        tokenized = self.tokenize(text)
        
        return tokenized

parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--batchsize', type=int, default=16)
arg('--n_chunks', type=int, default=8)
arg('--epochs', type=int, default=2)
arg('--accum', type=int, default=16)
arg('--maxlength', type=int, default=1024)
arg('--model', type=str, default='microsoft/deberta-v3-base')
arg('--learning_rate', type=float, default=5e-5)
#arg('--version', type=str, default='V05')
args = parser.parse_args()
print(args)

'''
# Download the below dataset to folder : `datamount/openwebtext2/chunks/`
#   --> https://huggingface.co/datasets/RaiBP/openwebtext2-first-30-chunks-english-only-examples
CHUNKS = sorted(glob.glob('datamount/openwebtext2/chunks/*.parquet'))
data = pd.concat([pq.read_table(fp).to_pandas() for fp in  \
                  tqdm(CHUNKS[:args.n_chunks], total = len(CHUNKS[:args.n_chunks]) )])
'''

data_path = "datamount/persuade_2.0_human_scores_demo_id.csv.gz"
data = pd.read_csv(data_path)


#data_path = "datamount/persuade_2.0_human_scores_demo_id.csv.gz"
#data = pd.read_csv(data_path)

texts = data["text"]
del data
gc.collect()
print('To lower')
texts = texts.str.lower()
print('update breaks')
texts = texts.str.replace('\n', '|')
print('change to `i`')
texts = texts.str.replace(r'[a-z0-9]', 'i', regex=True)
print(f'convert to list : {len(texts)}')
texts = texts.tolist()
print('preprocessing done.')

# model_name_or_path = "microsoft/deberta-v3-base"
# max_length = 1024
model_name_or_path = args.model
max_length = args.maxlength

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

dataset = PretrainingDataset(
    texts=texts, 
    tokenizer=tokenizer, 
    max_length=max_length,
)

#valdataset = PretrainingDataset(
#    texts=[t for i,t in enumerate(texts) if i%100==0], 
#    tokenizer=tokenizer, 
#    max_length=max_length,
#)

model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15,
)
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=f"weights/pretrain_mlm_v06k/{args.model}",
    fp16=True,
    lr_scheduler_type="cosine",
    weight_decay=0.005,
    warmup_ratio=0.02,
    learning_rate=args.learning_rate,
    gradient_accumulation_steps=args.accum,
    gradient_checkpointing=True,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batchsize,
    per_device_eval_batch_size=args.batchsize,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    #evaluation_strategy = 'steps'
    #eval_steps = 50, 
    report_to="wandb",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    #eval_dataset=valdataset,
    #compute_metrics=compute_metrics
)

trainer.train()
