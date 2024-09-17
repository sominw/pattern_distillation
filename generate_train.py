import pandas as pd
import transformers
import torch
import jsonlines
import json
import os
from datasets import load_dataset
import argparse
import click
import backoff
# from itertools import chain
# from openai import OpenAI
from together import Together
import together 
import csv
import numpy as np


def load_text_data(filepath, ckpt_ids, hf=False):
    if not hf:
        assert filepath.endswith('.csv')
        return pd.read_csv(filepath)  # Corrected variable name from 'filename' to 'filepath'
    elif hf: 
        if ckpt_ids:
            return load_dataset(filepath, '3.0.0', split='train', cache_dir='/scratch/shaib.c/').filter(lambda x: x['id'] in ids).to_pandas()
        else: 
            return load_dataset(filepath, '3.0.0', split='train', cache_dir='/scratch/shaib.c/').to_pandas()


@backoff.on_exception(backoff.expo,
                      (together.error.APIError,
                      together.error.RateLimitError,
                      together.error.APIConnectionError),
                      giveup=together.error.InvalidRequestError)

def call_together_api(text, model, key, client):
    PROMPT = "\n\nPlease summarize the given text. Be concise and respond only in 3-5 sentences."
    if text == np.nan:
        return 'nan'
    if text.startswith('['):
        text = str(text).lstrip('[').rstrip(']')
    try: 
        response = client.chat.completions.create(
            model=model,
            messages= [{"role": "user", "content": text + PROMPT}],
        )
    # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except: 
        return "invalid string, skipped."


def generate_summary(text, pipeline):
    PROMPT = "\n\nPlease summarize the given text."
    messages = [
        # {"role": "system", "content": "You are a professional, accurate summary writer, and you are not too verbose."},
        {"role": "user", "content": text + PROMPT},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=512
    )
    # print(outputs[0]["generated_text"][-1]['content'])
    return outputs[0]["generated_text"][-1]['content']


def save_summaries(summaries, filepath, idx):
    with jsonlines.open(filepath, mode='a') as writer:
        writer.write_all(summaries)
    print(f"Summaries have been saved to {filepath} at index {idx}")


def load_existing_ids(filepath):
    existing_ids = set()
    if os.path.exists(filepath):
        with jsonlines.open(filepath, mode='r') as reader:
            for obj in reader:
                existing_ids.add(obj['id'])
    return existing_ids


def generate_train_data(input_fp, hf, output_jsonl, save_interval, text_col, id_col, summary_col, ckpt_ids, model_id, checkpoint_fp=None, together_api=False):
    data = load_text_data(input_fp, ckpt_ids, hf)
    data_json = [] 

    if id_col == 'None': 
        data['id'] = data.index
        id_col = 'id'

    existing_ids = set()
    if checkpoint_fp:
        existing_ids = load_existing_ids(checkpoint_fp)
        print(f"Loaded {len(existing_ids)} existing summaries from {checkpoint_fp}")

    if not together_api: 
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    for idx, row in data.iterrows(): 
        if row[id_col] in existing_ids:
            # print(f"Skipping already processed ID: {row['id']}")
            continue
         
        if not together_api:
            processed_row = {
                "id": row[id_col], 
                "text": row[text_col],
                "gold_summary": row[summary_col], 
                "generated_summary": generate_summary(row[text_col], pipeline)
            }
        else: 
            key = open('/home/shaib.c/pattern_distillation/.togetherai_api_key.txt').read().strip()
            client = Together(api_key=key)
            processed_row = {
                "id": row[id_col], 
                "text": row[text_col],
                "gold_summary": row[summary_col], 
                "generated_summary": call_together_api(row[text_col], model_id, key, client)
            }

        data_json.append(processed_row)

        if idx % save_interval == 0 and data_json: 
            save_summaries(data_json, output_jsonl, idx)
            data_json = []

    if data_json:
        save_summaries(data_json, output_jsonl, idx)
    print("All summaries have been saved.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate training data')
    
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--together_api', type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    configs = json.load(open('config.json'))
    configs = {conf['name'] : conf for conf in configs}
    ids = None

    if args.dataset == 'cnn_dailymail': 
        hf = True 
        id_path = '/home/shaib.c/pattern_distillation/generated_data/cnn_dailymail_ids.txt'
        ids = open(id_path).read().splitlines()
    else:
        hf = False
    

    DATA_ROOT = '/work/frink/shaib.c/pattern_distillation/generated_data/'
    input_fp = configs['generation_' + args.dataset]['path']
    output_fp = DATA_ROOT + args.dataset + "_generated_" +  args.model_id.split('/')[-1]+".jsonl"

    save_interval = 100

    text_col = configs['generation_' + args.dataset]['text_col']
    id_col = configs['generation_' + args.dataset]['id_col']
    summary_col = configs['generation_' + args.dataset]['summary_col']
    
    checkpoint_fp = output_fp 
 
    generate_train_data(input_fp, hf, output_fp, save_interval, text_col, id_col, summary_col, ids, args.model_id, checkpoint_fp, args.together_api)