import os
import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer
import pandas as pd
import nltk
from tqdm import tqdm
import random
import spacy
import re

def get_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

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

def prepare_compute_metrics_clm(tokenizer: AutoTokenizer, rationales: bool = True): 
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred

        print (predictions.shape, labels.shape)
        predictions = np.argmax(predictions, axis = -1)

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        

        rand_ix = random.randint(0, len(decoded_labels) - 1)

        print ("\nLABELS (FULL): ", decoded_labels[rand_ix])
        print ("\nPREDS (FULL): ", decoded_predictions[rand_ix])

        result = rouge.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics