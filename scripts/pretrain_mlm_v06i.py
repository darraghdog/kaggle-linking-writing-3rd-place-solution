from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import warnings
import os
from transformers import AutoModelForMaskedLM, TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling, AutoTokenizer
import string
import argparse

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

print('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--batchsize', type=int, default=8)
arg('--accum', type=int, default=16)
arg('--epochs', type=int, default=25)
arg('--maxlength', type=int, default=1024)
arg('--model', type=str, default='microsoft/deberta-v3-large')
arg('--replace_char', type=str, default='i')
arg('--learning_rate', type=float, default=5e-5)
#arg('--version', type=str, default='V05')
args = parser.parse_args()
print(args)


data_path = "datamount/persuade_2.0_human_scores_demo_id.csv.gz"
data = pd.read_csv(data_path)

texts = data["full_text"]
texts = texts.str.lower()
texts = texts.str.replace('\n', '|')
for L in string.ascii_lowercase:
    texts = texts.str.replace(L, args.replace_char)
for L in range(10):
    texts = texts.str.replace(str(L), args.replace_char)

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

model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
model.config

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir=f"weights/pretrain_mlm_v06i/{args.model}",
    fp16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    learning_rate=args.learning_rate,
    gradient_accumulation_steps=args.accum,
    gradient_checkpointing=True,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batchsize,
    # per_device_eval_batch_size=32,
    save_steps=100,
    save_total_limit=2,
    # evaluation_strategy = 'steps'
    logging_steps=10,
    report_to="wandb",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
