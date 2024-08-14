import pandas as pd
import transformers
import torch
import jsonlines
import os
from datasets import load_dataset

def load_text_data(filepath, hf=False):
    if not hf:
        assert filepath.endswith('.csv')
        return pd.read_csv(filepath)  # Corrected variable name from 'filename' to 'filepath'
    elif hf: 
        return load_dataset(filepath, '3.0.0', split='train', cache_dir='/scratch/shaib.c/').to_pandas()

    
def generate_summary(text, pipeline):
    PROMPT = "\n\nPlease summarize the given text."
    messages = [
        {"role": "system", "content": "You are a professional, accurate summary writer, and you are not too verbose."},
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


def generate_train_data(input_fp, hf, output_jsonl, save_interval, text_col, model_id, checkpoint_fp=None):
    data = load_text_data(input_fp, hf)
    data_json = [] 

    existing_ids = set()
    if checkpoint_fp:
        existing_ids = load_existing_ids(checkpoint_fp)
        print(f"Loaded {len(existing_ids)} existing summaries from {checkpoint_fp}")


    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for idx, row in data.iterrows(): 
        if row['id'] in existing_ids:
            # print(f"Skipping already processed ID: {row['id']}")
            continue

        processed_row = {
            "id": row['id'], 
            "text": row[text_col],
            "gold_summary": row['highlights'], 
            "generated_summary": generate_summary(row[text_col], pipeline)
        }

        data_json.append(processed_row)

        if idx % save_interval == 0 and data_json: 
            save_summaries(data_json, output_jsonl, idx)
            data_json = []

    if data_json:
        save_summaries(data_json, output_jsonl, idx)
    print("All summaries have been saved.")


if __name__ == "__main__":
    input_fp = "ccdv/cnn_dailymail"
    hf = True
    model_id = "mistralai/Mistral-7B-Instruct-v0.3" # "google/gemma-2-9b-it" # "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    output_fp = "cnn_dailymail_generated_"+model_id.split('/')[-1]+".jsonl"
    save_interval = 100
    text_col = "article"
    
    checkpoint_fp = output_fp 
 
    generate_train_data(input_fp, hf, output_fp, save_interval, text_col, model_id)
    