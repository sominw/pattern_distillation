import os

DATASETS = {
    'rotten_tomatoes': '/work/frink/shaib.c/pattern_distillation/inference/rotten_tomatoes/',
    'pubmed': '/work/frink/shaib.c/pattern_distillation/inference/pubmed/',
    'cnn': '/work/frink/shaib.c/pattern_distillation/inference/cnn/',
    'alpaca': '/work/frink/shaib.c/pattern_distillation/inference/alpaca/'
}

MODELS = [
    "gpt2_gemma2_9b.csv", 
    "gpt2_llama8b.csv", 
    "gpt2_mixtral.csv", 
    "gpt2_llama70b.csv", 
    "gpt2_mistral7b.csv",
]

QA_DATASETS = {
    'commonsenseqa': '/work/frink/shaib.c/pattern_distillation/inference/qa_outputs/commonsenseqa_',
    'openbookqa': '/work/frink/shaib.c/pattern_distillation/inference/qa_outputs/openbookqa_',
    'quarel': '/work/frink/shaib.c/pattern_distillation/inference/qa_outputs/quarel_',
}

QA_TEACHERS = {
    # 'gemma2_9b_with_gold_rationales.csv', 
    'llama3_70b_with_gold_rationales.csv',
    'llama3_8b_with_gold_rationales.csv',
    'mistral_7b_with_gold_rationales.csv',
    'mixtral_with_gold_rationales.csv'
}

QA_STUDENTS = {
    # 'GPT2_gemma2_9b.csv',
    'GPT2_llama3_70b.csv',
    'GPT2_llama3_8b.csv',
    'GPT2_mistral7b.csv',
    'GPT2_mixtral.csv'  
}


MAX_FEATURES = 10000
NGRAM_RANGE = (1, 3)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000