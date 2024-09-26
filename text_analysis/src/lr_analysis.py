import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import logging
from utils.preprocessing import preprocess_text, truncate_text, get_word_count
from config import MAX_FEATURES, NGRAM_RANGE, TEST_SIZE, RANDOM_STATE, MAX_ITER, MODELS
from itertools import combinations 

def perform_lr_bow(df, text_column, label_column, truncate):
    if text_column not in df.columns or label_column not in df.columns:
        logging.error(f"Error: One or both columns ({text_column}, {label_column}) not found in the DataFrame.")
        return None

    
    df[text_column] = df[text_column].apply(preprocess_text)
    
    if truncate:
        min_words = df[text_column].apply(get_word_count).min()
        df[text_column] = df[text_column].apply(lambda x: truncate_text(x, min_words))
    
    X, y = df[text_column], df[label_column]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    try:
        vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        X = vectorizer.fit_transform(X)
        
        # Debug: Log vocabulary size and sample of features
        vocab_size = len(vectorizer.vocabulary_)
        logging.debug(f"Vocabulary size: {vocab_size}")
        if vocab_size > 0:
            sample_features = list(vectorizer.vocabulary_.keys())[:10]
            logging.debug(f"Sample features: {sample_features}")
        else:
            logging.error("Vocabulary is empty. Check the preprocessing steps and input data.")
        
        if X.shape[1] == 0:
            logging.error("CountVectorizer produced an empty feature set. Check your data for empty strings or only stop words.")
            return None
    except ValueError as e:
        logging.error(f"Error in CountVectorizer: {e}")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    for key in report:
        if isinstance(report[key], dict):
            report[key] = {k: round(v, 3) if isinstance(v, float) else v for k, v in report[key].items()}
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }

def pairwise_lr_analysis(dfs, text_column, label_column, analysis_name, truncate):
    results = []
    for (i, df1), (j, df2) in combinations(enumerate(dfs), 2):
        model1_data = df1[[text_column]].copy()
        model1_data['class'] = MODELS[i].split('.')[0]
        model2_data = df2[[text_column]].copy()
        model2_data['class'] = MODELS[j].split('.')[0]
        
        combined_df = pd.concat([model1_data, model2_data], ignore_index=True)
        
        lr_results = perform_lr_bow(combined_df, text_column, 'class', truncate)
        if lr_results:
            results.append({
                'Model 1': MODELS[i].split('.')[0],
                'Model 2': MODELS[j].split('.')[0],
                'Accuracy': lr_results['accuracy'],
                'Classification Report': lr_results['classification_report']
            })
    
    return pd.DataFrame(results)

def student_teacher_pairwise_lr(dfs, truncate):
    results = []
    for (i, student_df), (j, teacher_df) in combinations(enumerate(dfs), 2):
        student_data = student_df[['student']].copy()
        student_data['class'] = 'student'
        student_data.columns = ['text', 'class']
        
        teacher_data = teacher_df[['teacher_summ']].copy()
        teacher_data['class'] = 'teacher'
        teacher_data.columns = ['text', 'class']
        
        combined_df = pd.concat([student_data, teacher_data], ignore_index=True)
        
        lr_results = perform_lr_bow(combined_df, 'text', 'class', truncate)
        if lr_results:
            results.append({
                'Student Model': MODELS[i].split('.')[0],
                'Teacher Model': MODELS[j].split('.')[0].lower().split('gpt2_')[1],
                'Accuracy': lr_results['accuracy'],
                'Classification Report': lr_results['classification_report']
            })
    
    return pd.DataFrame(results)

def analyze_data(text_column, label_column, analysis_name, dfs, truncate):
    logging.info(f"\nDiscriminating between {analysis_name}:")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    results = perform_lr_bow(combined_df, text_column, label_column, truncate)
    
    if results:
        logging.info(f"Accuracy: {results['accuracy']:.3f}")
        logging.info(f"Classification Report for {analysis_name}:")
        logging.info(pd.DataFrame(results['classification_report']).round(3).transpose())
    else:
        logging.error(f"Analysis for {analysis_name} could not be performed due to data issues.")
    
    return results