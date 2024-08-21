from argparse import ArgumentParser, BooleanOptionalAction
import warnings
import json
import os
import torch
import logging
import random
import time
import numpy as np
import pandas as pd
import pickle as pk
import logging.config
import re
import datetime as dt
from tqdm import tqdm

import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer 
from load_data import load_data_for_inference

warnings.filterwarnings("ignore")

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
    LOG_FILE = LOG_FILE + "/" + args.data + "_benchmarked_" + args.student + "_" + args.teacher + "_" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S') + ".log"
    logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
    fh = logging.FileHandler("{0}".format(LOG_FILE))
    fh.setFormatter(logFormatter)
    logger = logging.getLogger('simple')
    logger.addHandler(fh)
    fh.setLevel(logging.DEBUG)

    set_seed(123)

    logger.info(args)
    cache_path = args.cache_path

    assert os.path.exists(cache_path), "Cache path must be reconfigured via --cache_path"
    m = args.model
    logger.info("Model Checkpoint: " + args.model)
    logger.info("Data: \t" + args.data)
    logger.info("Teacher: \t" + args.teacher)
    logger.info("Student: \t" + args.student)

    rouge = evaluate.load("rouge")

    if not args.save:
        logger.warning("Model generated outputs will NOT be saved!!")

    if args.verbose:
        logger.warning("Verbose is set to TRUE, every generated output & its corresponding references will be printed!!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info("# of GPUs in use: " + str(torch.cuda.device_count()))
    
    configs = json.load(open('../config.json'))
    configs = {conf['name'] : conf for conf in configs}
    assert args.data in set(configs.keys()), "Unavailable data, check config.json"
    config = configs[args.data]

    tokenizer = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, 
                                                cache_dir=cache_path,
                                                local_files_only = True,
                                                device_map="auto")
    
    d_test = load_data_for_inference(config, args.teacher, tokenizer, args.samples)
    logger.info("Data Loaded: " + config["name"] + " with " + str(len(d_test)) + " test instances")

    generated, gold, teacher, ip, ids = [], [], [], [], []
    erroneous = 0
    for ins in tqdm(d_test):
        m_input = ins["clm"]
        inputs = tokenizer(m_input, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(inputs, 
                            max_length=1024, 
                            # do_sample=True, 
                            # top_k=50, 
                            # top_p=0.95, 
                            # temperature=0.7,
                            num_return_sequences=1,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                            # eos_token_id=tokenizer.eos_token_id,
                            # bos_token_id=tokenizer.bos_token_id,
                            # no_repeat_ngram_size=2,
                            # early_stopping=True,
                            # num_beams=5,
                            # length_penalty=1.0,
                            )
        torch.cuda.empty_cache()
        generated_ids = outputs.to('cpu')
        generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        try:
            match = re.search(r'\[SUMMARY\]\s*(.*?)\s*\[SUMMARY\]', generated_tokens.strip())
            if match:
                summ = match.group(1)
                generated.append(summ)
            else:
                summ = "None"
                generated.append(summ)
                erroneous += 1
            gold.append(ins["gold_summary"])
            teacher.append(ins["generated_summary"])
            ip.append(ins["text"])
            ids.append(ins["id"])
            if args.verbose:
                print ("\n\nID: ", ins["id"])
                print ("\nTEXT: ", ins["text"])
                print ("\nSTUDENT GENERATED: ", summ)
                print ("\nGOLD: ", ins["gold_summary"])
                print ("\nTEACHER GENERATED: ", ins["generated_summary"])
                print ("\n\n---------------------------------------------\n\n")
        except:
            logger.error("Error in processing instance: " + ins["id"])
            erroneous += 1
    
    result = rouge.compute(predictions=generated, references=teacher, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in generated]
    result["gen_len"] = np.mean(prediction_lens)
    
    final_res = {k: round(v, 4) for k, v in result.items()}

    if args.save:
        path = "/work/frink/shaib.c/pattern_distillation/inference/" + args.data + "/"
        os.makedirs(path, exist_ok = True)
        pd.DataFrame({"id": ids, "text": ip, "student": generated, "teacher_summ": teacher, "gold_reference": gold}).to_csv(path + str(args.student + "_" + args.teacher +".csv"), index = False)
        logger.info("Augmented data with outputs saved at: " + path)

    logger.info("Final Results: \n" + str(final_res))
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required = True)
    parser.add_argument("--model", type=str, required = True)
    parser.add_argument("--teacher", type=str, required = True)
    parser.add_argument("--student", type=str, required = True)
    parser.add_argument("--cache_path", type=str, default="/scratch/wadhwa.s/pattern_distillation/")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--save", action=BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=BooleanOptionalAction, default=False)

    args_lightning = parser.parse_args()
    main(args_lightning)
