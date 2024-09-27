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
from sklearn.feature_extraction.text import TfidfVectorizer

def perform_lr_bow(df, text_column, label_column, truncate):

    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    if truncate:
        df['processed_text'] = df['processed_text'].apply(lambda x: truncate_text(x, 100))
    
    # vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X = vectorizer.fit_transform(df['processed_text'])
    
    le = LabelEncoder()
    y = le.fit_transform(df[label_column])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    clf = LogisticRegression(multi_class='multinomial', random_state=RANDOM_STATE, max_iter=MAX_ITER)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    return accuracy, report, clf, vectorizer, le


def multiclass_lr_teacher_classification(dfs, truncate=False):

    combined_df = pd.concat(dfs, ignore_index=True)

    accuracy, report, clf, vectorizer, le = perform_lr_bow(combined_df, 'student', 'model', truncate)
    
    logging.info(f"Multiclass LR Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{pd.DataFrame(report).transpose()}")

    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out(),
        'importance': clf.coef_.mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(20)
    logging.info(f"Top 20 important features:\n{top_features}")
    
    return accuracy, report, feature_importance