import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
import warnings
import json
import torch
import logging
import random
import numpy as np
import time
import pandas as pd
import datetime as dt
import logging.config
from datasets import Dataset
# from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from trl import DataCollatorForCompletionOnlyLM

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_preprocess_clm_fn(tokenizer: AutoTokenizer):
    def preprocess_fn(instances):
        # print (instances["clm"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer.pad_token = "[PAD]"
        model_inputs = tokenizer(instances["clm"], 
                                max_length=512, 
                                truncation=True, 
                                padding=True,  
                                return_tensors="pt",
                                ).to(device)
        return model_inputs

    return preprocess_fn

set_seed(123)

warnings.filterwarnings("ignore")

cache_path = "/scratch/wadhwa.s/pattern_distillation/"

m = "openai-community/gpt2"

model = AutoModelForCausalLM.from_pretrained(m, 
                                                cache_dir=cache_path, 
                                                device_map = "auto"
                                                )
tokenizer = AutoTokenizer.from_pretrained(m, 
                                            cache_dir=cache_path,
                                            # padding_side='left',
                                            use_fast=False)

data_path = "/work/frink/shaib.c/pattern_distillation/generated_data/"
cnn_dailymail_llama8b = os.path.join(data_path, "cnn_dailymail_generated_Mistral-7B-Instruct-v0.3.jsonl")

data = cnn_dailymail_llama8b

with open(data, 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df = df.astype(str) 
df.shape

df["clm"] = tokenizer.bos_token + df["text"] + "#### [SUMMARY]" + df["generated_summary"] + "[SUMMARY]"

df = df.sample(1000)

d = Dataset.from_pandas(df)
d = d.train_test_split(test_size=0.2)
d_train = d['train']
d_valid = d['test']

preprocess_fn = prepare_preprocess_clm_fn(tokenizer)

tokenized_d_train = d_train.map(preprocess_fn, batched=True)
tokenized_d_valid = d_valid.map(preprocess_fn, batched=True)

tokenized_d_train = tokenized_d_train.remove_columns(d_train.column_names)
tokenized_d_valid = tokenized_d_valid.remove_columns(d_valid.column_names)

output_path = cache_path + "pattern_distill_models/"

print ("OUTPUT PATH: ", output_path)

response_template = "#### [SUMMARY]"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
                # report_to=args.report_to,
                output_dir=output_path,
                evaluation_strategy="steps",    
                eval_steps=100,
                learning_rate=3e-6,
                save_strategy="steps",
                save_steps=100,
                per_device_train_batch_size=12,
                per_device_eval_batch_size=12,
                auto_find_batch_size=False,
                eval_delay=200,
                logging_strategy="steps",
                logging_steps=100,
                weight_decay=0.01,
                save_total_limit=10,
                num_train_epochs=30,
                logging_dir=output_path + "/logs",
                load_best_model_at_end = True,
                metric_for_best_model = "eval_loss",
                greater_is_better = False,
                # log_level="info",
                eval_accumulation_steps=10,
                gradient_accumulation_steps=2
                )

trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_d_train,
                eval_dataset=tokenized_d_valid,
                tokenizer=tokenizer,
                # compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02)]
                )

trainer.train()