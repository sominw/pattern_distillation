import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rouge import Rouge
from utils.preprocessing import preprocess_text, truncate_text, get_word_count

def calculate_similarity(text1, text2, truncate):
    text1, text2 = map(preprocess_text, (text1, text2))
    
    if truncate:
        min_words = min(get_word_count(text1), get_word_count(text2))
        text1 = truncate_text(text1, min_words)
        text2 = truncate_text(text2, min_words)
    
    if not text1 or not text2:
        return {metric: 0 for metric in ['cosine_similarity', 'rouge_1_f', 'rouge_2_f', 'rouge_l_f']}
    
    try:
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(bow[0], bow[1])[0][0]
    except ValueError:
        cosine_sim = 0
    
    # try:
    #     bleu = sentence_bleu([word_tokenize(text1)], word_tokenize(text2))
    # except ValueError:
    #     bleu = 0
    
    try:
        rouge = Rouge()
        rouge_scores = rouge.get_scores(text1, text2)[0]
    except ValueError:
        rouge_scores = {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
    
    return {
        'cosine_similarity': round(cosine_sim, 3),
        # 'bleu_score': round(bleu, 3),
        'rouge_1_f': round(rouge_scores['rouge-1']['f'], 3),
        'rouge_2_f': round(rouge_scores['rouge-2']['f'], 3),
        'rouge_l_f': round(rouge_scores['rouge-l']['f'], 3)
    }