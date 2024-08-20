from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoTokenizer

def load_data_for_training_evaluation(config: dict, 
                                      train: bool = True, 
                                      teacher: str = "mistral",
                                      tokenizer: AutoTokenizer = None):
    """
    Load data for text2text model training
    """
    df = pd.read_csv(config[teacher+"_filtered.csv"])
    df["clm"] = tokenizer.bos_token + df["text"] + " #### [SUMMARY]" + df["generated_summary"] + "[SUMMARY]"
    d = Dataset.from_pandas(df)
    d = d.train_test_split(test_size=0.2)


    return d['train'], d['test'], d['test']