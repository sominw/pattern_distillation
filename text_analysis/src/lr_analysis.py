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

def perform_lr_bow(df, original_df, text_column, label_column, teacher_summ, truncate):

    # df --> students only 
    # original_/train_df --> teachers only
    print(len(df), len(original_df))
    df['processed_text'] = df[text_column].apply(preprocess_text)
    original_df['processed_text'] = original_df[teacher_summ].apply(preprocess_text)

    # sample 20-50 from each model type in original_df, making sure they are different
    # only works for when sample size is smaller than the original df!
    train_df = pd.DataFrame()
    for model in df['model'].unique():
        df_to_sample_from = original_df[~(original_df['id'].isin(df['id']))]
        model_df = df_to_sample_from[df_to_sample_from['model'] == model]# .sample(50, random_state=RANDOM_STATE)
        train_df = pd.concat([train_df, model_df], ignore_index=True)

    if truncate:
        df['processed_text'] = df['processed_text'].apply(lambda x: truncate_text(x, 100))
    
    # vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    le = LabelEncoder()

    X_train, y_train = vectorizer.fit_transform(train_df['processed_text']), le.fit_transform(train_df['model'])
    X_test, y_test = vectorizer.transform(df['processed_text']), le.transform(df['model'])


    # clf = LogisticRegression(multi_class='ovr', random_state=RANDOM_STATE, max_iter=MAX_ITER)
    clf = LogisticRegression(multi_class='multinomial', random_state=RANDOM_STATE, max_iter=MAX_ITER)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=df['model'].unique(), output_dict=True)

    return accuracy, report, clf, vectorizer, le


def multiclass_lr_teacher_classification(dfs, original_df, truncate=False):

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_original_df = pd.concat(original_df, ignore_index=True)

    accuracy, report, clf, vectorizer, le = perform_lr_bow(combined_df, combined_original_df, 'student', 'model', 'teacher_summ', truncate)
    
    logging.info(f"Multiclass LR Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{pd.DataFrame(report).transpose()}")

    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out(),
        'importance': clf.coef_.mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(20)
    # logging.info(f"Top 20 important features:\n{top_features}")
    
    return accuracy, report, feature_importance