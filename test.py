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
import evaluate
import re
import logging.config
from datasets import Dataset
# from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    logits, labels = torch.Tensor(logits), torch.Tensor(labels)
    logits = torch.argmax(logits, dim = -1)

    shifted_labels = torch.roll(labels, -1, dims = 1)
    shifted_labels[:, -1] = -100 # dummy value

    pruned = [logit[label != -100] for logit, label in zip(logits, shifted_labels)]
    max_size = max([prune.shape[0] for prune in pruned])

    new_logits = [torch.cat((logit, torch.full((max_size - logit.shape[0],), -100, device=logit.device)), dim=0) for logit in pruned]
    new_logits = torch.stack(new_logits, dim=0)

    return new_logits

def prepare_compute_metrics_clm(tokenizer: AutoTokenizer, rationales: bool = True): 
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred
        # predictions = preprocess_logits_for_metrics(predictions, labels)

        # if isinstance(predictions, tuple):
        #     predictions = predictions[0]

        # print (type(predictions), type(labels))
        print (predictions.shape, labels.shape)
        # print (labels[0])
        predictions = np.argmax(predictions, axis = -1)

        # predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        

        rand_ix = random.randint(0, len(decoded_labels) - 1)

        print ("\nLABELS (FULL): ", decoded_labels[rand_ix])
        print ("\nPREDS (FULL): ", decoded_predictions[rand_ix])

        # results = { "accuracy": accuracy.compute(predictions=normalize_to_ints(decoded_predictions), references=normalize_to_ints(decoded_labels))["accuracy"] }
        result = rouge.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics

def prepare_preprocess_clm_fn(tokenizer: AutoTokenizer):
    def preprocess_fn(instances):
        # print (instances["clm"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer.pad_token = "[PAD]"
        model_inputs = tokenizer(instances["clm"], 
                                max_length=1024, 
                                truncation=True, 
                                padding=True,  
                                return_tensors="pt",
                                ).to(device)
        return model_inputs

    return preprocess_fn

set_seed(100)

warnings.filterwarnings("ignore")

cache_path = "/work/frink/wadhwa.s/pattern_distillation/"

m = "openai-community/gpt2-xl"

model = AutoModelForCausalLM.from_pretrained(m, 
                                                cache_dir=cache_path, 
                                                device_map = "auto",
                                                torch_dtype = torch.float32,
                                                )
tokenizer = AutoTokenizer.from_pretrained(m, 
                                            cache_dir=cache_path,
                                            # padding_side='left',
                                            use_fast=False)

data_path = "/work/frink/shaib.c/pattern_distillation/generated_data/"
# cnn_dailymail_llama8b = os.path.join(data_path, "cnn_dailymail_generated_Mistral-7B-Instruct-v0.3.jsonl")

# data = cnn_dailymail_llama8b

# with open(data, 'r') as f:
#     data = [json.loads(line) for line in f]

# df = pd.DataFrame(data)
# df = df.astype(str) 
# df.shape

df = pd.read_csv("/work/frink/shaib.c/pattern_distillation/generated_data/cnn_dailymail_generated_Mistral-7B-Instruct-v0.3_filtered.csv")

df["clm"] = tokenizer.bos_token + df["text"] + " #### [SUMMARY]" + df["generated_summary"] + "[SUMMARY]"

# df = df.sample(1000)

d = Dataset.from_pandas(df)
d = d.train_test_split(test_size=0.2)
d_train = d['train']
d_valid = d['test']

preprocess_fn = prepare_preprocess_clm_fn(tokenizer)

tokenized_d_train = d_train.map(preprocess_fn, batched=True)
tokenized_d_valid = d_valid.map(preprocess_fn, batched=True)

# print ("\n\nTRAIN SAMPLE", tokenized_d_train[0])
# print ("\n\nVALID SAMPLE", tokenized_d_valid[0])

tokenized_d_train = tokenized_d_train.remove_columns(d_train.column_names)
tokenized_d_valid = tokenized_d_valid.remove_columns(d_valid.column_names)

output_path = cache_path + "pattern_distill_models/"
# print (d_train[:1])
# print ("\n\nTRAIN SAMPLE", tokenized_d_train[0])
# print ("\n\nVALID SAMPLE", tokenized_d_valid[0])

# print ("TRAIN SHAPE: ", tokenized_d_train.shape)
# print ("VALID SHAPE: ", tokenized_d_valid.shape)
# print (tokenized_d_train.shape, tokenized_d_valid.shape)
print ("OUTPUT PATH: ", output_path)

response_template = " #### [SUMMARY]"
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
compute_metrics = prepare_compute_metrics_clm(tokenizer)

training_args = TrainingArguments(
                report_to="none",
                output_dir=output_path,
                evaluation_strategy="steps",    
                eval_steps=500,
                learning_rate=3e-6,
                save_strategy="steps",
                save_steps=500,
                per_device_train_batch_size=12,
                per_device_eval_batch_size=12,
                auto_find_batch_size=False,
                eval_delay=500,
                logging_strategy="steps",
                logging_steps=500,
                weight_decay=0.01,
                save_total_limit=10,
                num_train_epochs=30,
                logging_dir=output_path + "/logs",
                load_best_model_at_end = True,
                metric_for_best_model = "eval_loss",
                greater_is_better = False,
                # log_level="info",
                eval_accumulation_steps=10,
                # gradient_accumulation_steps=2
                )

trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_d_train,
                eval_dataset=tokenized_d_valid,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02)]
                )

trainer.train()