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
# from transformers.utils import logging
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

def main(args):
    
    logging.config.fileConfig('logging.conf')
    LOG_FILE = os.getcwd() + "/logs"
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)

    if args.train:
        LOG_FILE = LOG_FILE + "/" + "train_log_" + args.data + "_on_" + args.model + "_" + args.teacher + "_" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S') + ".log"

    logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
    fh = logging.FileHandler("{0}".format(LOG_FILE))
    fh.setFormatter(logFormatter)
    logger = logging.getLogger('simple')
    logger.addHandler(fh)
    fh.setLevel(logging.DEBUG)

    set_seed(123)
    logger.info (str(args))
    cache_path = args.cache_path
    assert os.path.exists(cache_path), "Cache path must be reconfigured via --cache_path"
    assert args.model in available_models.keys(), "Selected model must be from one of the available instruct models." 
    m = available_models[args.model]

    logger.info("Data: \t" + args.data)
    logger.info("Teacher Model: \t" + args.teacher)
    logger.info("Student Model: " + args.model + "\ti.e. " + m)

    configs = json.load(open('../config.json'))
    configs = {conf['name'] : conf for conf in configs}

    assert args.data in set(configs.keys()), "Unavailable data, check config.json"

    config = configs[args.data]

    

    model = AutoModelForCausalLM.from_pretrained(m, 
                                            cache_dir=cache_path, 
                                            device_map = "auto"
                                            )
    tokenizer = AutoTokenizer.from_pretrained(m, 
                                            cache_dir=cache_path,
                                            # padding_side='left',
                                            use_fast=False)
    additional_tokens = ['[SUMMARY]']
    special_tokens_dict = {'additional_special_tokens': additional_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = "[PAD]"

    compute_metrics = prepare_compute_metrics(tokenizer)

    d_train, d_valid, d_test = load_data_for_training_evaluation(config, args.train, args.teacher, tokenizer, args.sample_test)
    preprocess_fn = prepare_preprocess_clm_fn(tokenizer)

    tokenized_d_train = d_train.map(preprocess_fn, batched=True)
    tokenized_d_valid = d_valid.map(preprocess_fn, batched=True)

    # print (tokenized_d_train[1])

    # print ("\n\n")

    # print (tokenized_d_valid[1])

    tokenized_d_train = tokenized_d_train.remove_columns(d_train.column_names)
    tokenized_d_valid = tokenized_d_valid.remove_columns(d_valid.column_names)

    output_path = cache_path + "pattern_distill_models/"

    if args.output_path is None and args.train:
         output_path = output_path + "trained/" + args.data + "/" + args.model + "_" + args.teacher + "/"

    logger.info("Output Path: \t" + output_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
    # print (d_train[1])

    assert output_path.endswith("/"), "Supplied output path must be a directory ending wth '/'"

    if torch.cuda.is_available():
        logger.info("# of GPUs in use: " + str(torch.cuda.device_count()))

    training_args = TrainingArguments(
                    report_to=args.report_to,
                    output_dir=output_path,
                    evaluation_strategy=args.evaluation_strategy,    
                    eval_steps=args.eval_steps,
                    learning_rate=args.lr,
                    save_strategy=args.save_strategy,
                    per_device_train_batch_size=args.batch_size,
                    per_device_eval_batch_size=args.batch_size,
                    auto_find_batch_size=args.auto_find_batch_size,
                    eval_delay=args.eval_delay,
                    logging_strategy=args.logging_strategy,
                    logging_steps=args.logging_steps,
                    weight_decay=args.weight_decay,
                    save_total_limit=args.save_total_limit,
                    num_train_epochs=args.max_epochs,
                    logging_dir=output_path + "/logs",
                    load_best_model_at_end = args.load_best_model_at_end,
                    metric_for_best_model = args.metric_for_best_model,
                    greater_is_better = args.greater_is_better,
                    log_level="info",
                    eval_accumulation_steps=args.eval_accumulation_steps,
                    )
    
    trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=tokenized_d_train,
                    eval_dataset=tokenized_d_valid,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.02)]
                    )
    
    if args.train:
                trainer.train()

    print ("works until here!")
    logger.info("completed test")

if __name__ == "__main__":
    parser = ArgumentParser()

    # model_group = parser.add_mutually_exclusive_group(required=True)
    # model_group.add_argument("--model", type=str, default="gpt2_s")
    parser.add_argument("--data", type=str, default="cnn")
    parser.add_argument("--cache_path", type=str, default="/scratch/wadhwa.s/pattern_distillation/")
    parser.add_argument("--log_file", type=str, default="run.log")
    parser.add_argument("--log_level", type=str, default="INFO")

    # model training args (inc. HPs)
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=15)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--predict_with_generate", type=bool, default=True, action=BooleanOptionalAction)
    parser.add_argument("--eval_delay", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--auto_find_batch_size", default=False, action=BooleanOptionalAction)
    parser.add_argument("--eval_steps", type=int, default=500),
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True, action=BooleanOptionalAction)
    parser.add_argument("--greater_is_better", type=bool, default=False, action=BooleanOptionalAction)
    parser.add_argument("--generation_max_length", type=int, default=1024)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--eval_accumulation_steps", type=int, default=12)


    # src specific
    parser.add_argument("--teacher", type=str, default="mistral")
    parser.add_argument("--model", type=str, default="gpt2_s")
    parser.add_argument("--train", type=bool, default=True, action=BooleanOptionalAction)
    # parser.add_argument("--ablation", type=str, default = None)
    # parser.add_argument("--abl_param", type=float, default=0)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--sample_test", type=int, default=0)
    
    args_lightning = parser.parse_args() 
    main(args_lightning)