from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoTokenizer

def load_data_for_training_evaluation(config: dict, 
                                      train: bool = True, 
                                      teacher: str = "mistral7b",
                                      tokenizer: AutoTokenizer = None,
                                      sample_test: int = 0):
    """
    Load data for text2text model training
    """
    # print (config)
    df = pd.read_csv(config[teacher+"_filtered"])
    if sample_test > 0:
        df = df.sample(sample_test)
    df["clm"] = tokenizer.bos_token + df["text"] + " #### [SUMMARY]" + df["generated_summary"] + "[SUMMARY]"
    d = Dataset.from_pandas(df)
    d = d.train_test_split(test_size=0.2)

    d["test"] = d["test"].select(range(50))

    return d['train'], d['test'], d['test']

def load_data_for_inference(config: dict, 
                            teacher: str = "mistral7b",
                            tokenizer: AutoTokenizer = None,
                            sample_test: int = 0):
    """
    Load data for text2text model inference
    """
    # print (config)
    df = pd.read_csv(config[teacher+"_test"])
    if sample_test > 0:
        df = df.sample(sample_test)
    df["clm"] = tokenizer.bos_token + df["text"] + " #### [SUMMARY]"
    d = Dataset.from_pandas(df)
    return d