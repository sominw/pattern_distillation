import pandas as pd
from nltk.tokenize import word_tokenize
from config import DATASETS, MODELS, QA_DATASETS, QA_TEACHERS, QA_STUDENTS
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def preprocess_text(text):
    return ' '.join(word_tokenize(str(text).lower())) if pd.notna(text) else ""


def truncate_text(text, max_words):
    words = text.split()
    return ' '.join(words[:max_words])


def get_word_count(text):
    return len(word_tokenize(text))


def map_qa_dataset(teacher_path, student_path): 
    teacher = pd.read_csv(teacher_path)
    student = pd.read_csv(student_path)

    df = teacher.merge(student, on='id', suffixes=('_teacher', '_student'))

    df.rename(columns={'rationale': 'student', 'gold_rationale': 'teacher_summ', 'id_student': 'id'}, inplace=True)

    df = df[['id', 'student', 'teacher_summ']]
    return df

def load_dfs(dataset, truncate, same_ids, sample_size=None, qa_datasets=False):
    dfs = []
    all_ids = set()
    
    def process_df(df, model):
        df.dropna(subset=['student'], inplace=True)
        df['model'] = model
        df['length'] = df['student'].apply(lambda x: len(str(x).split()))
        logging.info(f"Average word length for model {model}: {df['length'].mean()}")
        return df
    
    if qa_datasets:
        for teacher, student in zip(QA_TEACHERS, QA_STUDENTS):
            try:
                teacher_fp = QA_DATASETS[dataset] + teacher
                student_fp = QA_DATASETS[dataset] + student
                model = student.split('.')[0]
                
                df = map_qa_dataset(teacher_fp, student_fp)
                df = process_df(df, model)
                
                if df.empty:
                    logging.warning(f"Empty DataFrame for {model} in {dataset}. Skipping.")
                    continue
                
                all_ids.update(df['id'])
                dfs.append(df)
            except FileNotFoundError:
                logging.warning(f"File {model} not found for {dataset}. Skipping.")
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file for {model} in {dataset}. Skipping.")
    else:
        for model in MODELS:
            try:
                df = pd.read_csv(os.path.join(DATASETS[dataset], model))
                df = process_df(df, model.split('.')[0])
                
                if df.empty:
                    logging.warning(f"Empty DataFrame for {model} in {dataset}. Skipping.")
                    continue
                
                all_ids.update(df['id'])
                dfs.append(df)
            except FileNotFoundError:
                logging.warning(f"File {model} not found for {dataset}. Skipping.")
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file for {model} in {dataset}. Skipping.")
    
    if not dfs:
        logging.error(f"No data files were successfully loaded for {dataset}.")
        return []
    
    if sample_size:
        if same_ids:
            # sample the same IDs across all dataframes
            sampled_ids = set(pd.Series(list(all_ids)).sample(min(sample_size, len(all_ids)), random_state=2024))
            dfs = [df[df['id'].isin(sampled_ids)] for df in dfs]
        else:
            # sample different IDs for each dataframe
            dfs = [df.sample(min(sample_size, len(df)), random_state=2024) for df in dfs]
    
    return dfs