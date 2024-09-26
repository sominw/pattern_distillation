from argparse import ArgumentParser, BooleanOptionalAction
import warnings
import json
import os
import torch
import logging
import random
import numpy as np
import time
import datetime as dt
import pandas as pd
import pickle as pk
import logging.config
import backoff
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer 
warnings.filterwarnings("ignore")
from together import Together
import together 

available_models = {
    "mistral_7b" : "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3_8b" : "meta-llama/Meta-Llama-3-8B",
    "gemma2_9b" : "google/gemma-2-9b-it",
    "llama3_70b" : "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mixtral" : "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

@backoff.on_exception(backoff.expo,
                      (together.error.APIError,
                      together.error.RateLimitError,
                      together.error.APIConnectionError),
                      giveup=together.error.InvalidRequestError)

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
    LOG_FILE = LOG_FILE + "/" + "_rationale_generation_" + args.model + "_" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S') + ".log"
    logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
    fh = logging.FileHandler("{0}".format(LOG_FILE))
    fh.setFormatter(logFormatter)
    logger = logging.getLogger('simple')
    logger.addHandler(fh)
    fh.setLevel(logging.DEBUG)

    set_seed(123)

    logger.info("Available Models: " + str(list(available_models.keys())))

    cache_path = args.cache_path

    # Model loading sanity checks
    assert os.path.exists(cache_path), "Cache path must be reconfigured via --cache_path"
    if args.load_from_ckpt is None:
        assert args.model in available_models.keys(), "Selected model must be from one of the available instruct models." 
        m = available_models[args.model]
    if args.load_from_ckpt is not None:
        logger.info("Model: " + args.load_from_ckpt)
        m = args.load_from_ckpt
    else:
        logger.info("Model: " + args.model + "\ti.e. " + m)
    
    logger.info("Data: \t" + args.data)
    remote_code = False

    if "falcon" in m:
        remote_code = True

    if not args.save:
        logger.warning("Augmented file with rationales will NOT be saved!!")
    
    if args.verbose:
        logger.warning("Verbose is set to TRUE, every generated output will be printed!!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        logger.info("# of GPUs in use: " + str(torch.cuda.device_count()))
    
    if args.save_dict:
        assert args.save == True, "Cannot save dictionary without saving the generated outputs"

    if args.togetherapi is None:
        tokenizer = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m, 
                                                    cache_dir=cache_path,
                                                    trust_remote_code = remote_code,
                                                    local_files_only = False,
                                                    device_map="auto")
    
    df = pd.read_csv(args.data)
    
    logger.info("Data Loaded: " + " with " + str(len(df)) + " examples")

    ids, processed_ids = [], []
    x, processed_x = [], []
    rationales = []
    output_directory = "/work/frink/wadhwa.s/pattern_distill_code/"
    os.makedirs(output_directory, exist_ok = True)

    if args.sample == 0:
        args.sample = df.shape[0]

    ctr = 0
    for ix, row in df.iterrows():
        if ctr == args.sample:
            break
        ids.append(row["id"])
        x.append(row["text"])
        ctr += 1

    with open(args.icl_prompt_file, "r") as f:
        prompt = f.read()

    for ix, (id, text) in tqdm(enumerate(zip(ids, x)), total = len(ids)):
        torch.cuda.empty_cache()
        m_input = prompt + "\n\n" + text + "\nAnswer:"
        
        if args.togetherapi is not None:
                key = args.togetherapi
                client = Together(api_key=key)
                # processed_row = {
                #     "id": row[id_col], 
                #     "text": row[text_col],
                #     "gold_summary": row[summary_col], 
                #     "generated_summary": call_together_api(row[text_col], model_id, key, client)
                # }
                try: 
                    response = client.chat.completions.create(
                        model=m,
                        messages= [{"role": "user", "content": m_input}],
                    )
                    out = response.choices[0].message.content
                    processed_ids.append(id)
                    processed_x.append(text)
                    # processed_y.append(label)
                    rationales.append(out)
                except: 
                    out = "invalid string, skipped."
        else:
            inputs = tokenizer(m_input, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(inputs, 
                            max_new_tokens = 128,
                            min_length = 25,
                            eos_token_id = tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            num_return_sequences= args.num_return_sequences,
                            #  num_beams= 3,    # uncomment if you have sufficient compute
                            do_sample = True,   # sampling set to True by default, can be changed later. 
                            top_p = 0.95,       # alter at own risk
                            top_k = 50,         # alter at own risk
                            #  max_length = 8 + len(inputs[0]),
                            #  pad_token_id = tokenizer.eos_token_id, # Mandatory for Falcon-style models
                            # stopping_criteria = TokenBasedStoppingCriteria("</s>", m_input, tokenizer),     # Not required .... or needs to be reconfigured for CausalLMs
                            return_dict_in_generate = args.save_dict,
                            output_scores = args.save_dict,
                            output_attentions = args.save_dict,
                            output_hidden_states = args.save_dict,
                            )
            torch.cuda.empty_cache()
            if args.save_dict:
                generated_ids = outputs['sequences'].to('cpu')
            else:
                generated_ids = outputs.to('cpu')
            generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            try:
                rationale = generated_tokens[0].split("\nAnswer: ")[-1].strip()
                processed_ids.append(id)
                processed_x.append(text)
                # processed_y.append(label)
                rationales.append(rationale)
                if args.save and args.save_dict:
                    output_dict = {
                        "id" : id,
                        "generated_sequences": generated_ids,
                        "scores": outputs["scores"],
                        "attentions": outputs["attentions"],
                        "hidden_states": outputs["hidden_states"],
                        "past_key_values": outputs["past_key_values"],
                    }
                    output_filename = os.path.join(output_directory, f'{id}.pkl')
                    with open(output_filename, 'wb') as f:
                        pk.dump(output_dict, f)
                    del outputs, output_dict
                out = rationale
            except:
                logger.error("Failed to generate rationale for: " + id)
                out = "No Rationale Generated"
                continue
        if args.verbose:
            print (text, "\nAnswer: " + out)
            if args.num_return_sequences > 1:
                for index, r in enumerate(generated_tokens):
                    print ("\nAnswer ", index, ":\t", r.split("\nAnswer: ")[-1].strip())
            print("\n----------------------------------------\n") 
        torch.cuda.empty_cache()

    if args.save:
        path = "/work/frink/wadhwa.s/pattern_distill_code/qa_outputs/" + args.base
        # os.makedirs(path, exist_ok = True)
        pd.DataFrame({"id": processed_ids, "text": processed_x, "gold_rationale": rationales}).to_csv(path + "_" + args.model + "_with_gold_rationales.csv", index = False)
        logger.info("Augmented data with rationales saved at: " + path)
        if args.save_dict:
            logger.info("Generated output rationale token IDs (num_generated_sequences per instance; token ids, attention scores etc) saved at: " + output_directory)
    
    # del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()

    model_group = parser.add_mutually_exclusive_group(required = True)    
    model_group.add_argument("--model", type=str, default = None)
    model_group.add_argument("--load_from_ckpt", type=str, default = None)
    parser.add_argument("--data", type=str, required = True)
    parser.add_argument("--cache_path", type=str, default = "/work/frink/wadhwa.s/.cache/")        # route to wherever you have a local copy of the models
    parser.add_argument("--log_file", type=str, default = "run.log")
    parser.add_argument("--log_level", type=str, default = "INFO")

    # ICL Specific 
    parser.add_argument("--icl_prompt_file", type=str, required=True)
    parser.add_argument("--prefix", type=str, default = "Given the following two examples of question-answer-rationale triplets, provide a rationale for the third example for why the selected choice answers the question.\n")      # reserved
    parser.add_argument("--postfix", type=str, default = "")      # reserved
    parser.add_argument("--num_return_sequences", type = int, default = 1)
    parser.add_argument("--num_beams", type=int, default = 1)

    # src specific
    parser.add_argument("--use_device", type=int, default = 0)                                     # reserved (can be used to parallelize if you have sufficient compute)
    parser.add_argument("--verbose", type=bool, default = False, action = BooleanOptionalAction)
    parser.add_argument("--save", type=bool, default = False, action = BooleanOptionalAction)
    parser.add_argument("--save_dict", type=bool, default = False, action = BooleanOptionalAction) # for saving scores, attention outputs, hidden states etc
    parser.add_argument("--sample", type=int, default = 0)                                           # reserved
    parser.add_argument("--rfact", type=bool, default = False, action = BooleanOptionalAction)
    parser.add_argument("--togetherapi", type=str, default = None)
    parser.add_argument("--base", type=str, default = None, required = True)

    args_lightning = parser.parse_args()
    main(args_lightning)