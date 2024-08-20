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
import datetime as dt
import logging.config
from datasets import Dataset
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling

from load_data import load_data_for_training_evaluation
from utils import prepare_compute_metrics_clm as prepare_compute_metrics, prepare_preprocess_clm_fn

warnings.filterwarnings("ignore")

available_models = {
    "gpt2_s" : "openai-community/gpt2",
    "gpt2_xl" : "openai-community/gpt2-xl",
    "phi" : "microsoft/phi-1_5",
}

dataset_map = {
    "cnn": "cnn_dailymail",
    "rotten_tomatoes": "rotten_tomatoes",
    "cochrane": "cochrane",
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)