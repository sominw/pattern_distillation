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

    # join on ID
    df = teacher.merge(student, on='id', suffixes=('_teacher', '_student'))

    # rename columns to id, student, teacher_summ
    df.rename(columns={'rationale': 'student', 'gold_rationale': 'teacher_summ', 'id_student': 'id'}, inplace=True)

    # drop other columns
    df = df[['id', 'student', 'teacher_summ']]
    df.dropna(subset=['student'], inplace=True)

    return df


def load_dfs(dataset, truncate, sample_size=None, qa_datasets=False):

    dfs = []

    if qa_datasets:
        for (teacher, student) in zip(QA_TEACHERS, QA_STUDENTS):
            try:
                # print(QA_DATASETS[dataset] + teacher)
                teacher_fp = QA_DATASETS[dataset] + teacher
                student_fp = QA_DATASETS[dataset] + student
                model = student.split('.')[0]

                df = map_qa_dataset(teacher_fp, student_fp) # should return a df that is [id text student teacher_summ]

                if sample_size:
                    df = df.sample(sample_size, random_state=2024)

                df['model'] = model
                df['length'] = df['student'].apply(lambda x: len(str(x).split()))
                
                logging.info(f"Average word length for model {model}: {df['length'].mean()}")
                
                if df.empty:
                    logging.warning(f"Empty DataFrame for {model} in {dataset}. Skipping.")
                    continue

                dfs.append(df)
            except FileNotFoundError:
                print('')
                logging.warning(f"File {model} not found for {dataset}. Skipping.")
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file for {model} in {dataset}. Skipping.")
    else: 
        for model in MODELS:
            try:
                df = pd.read_csv(os.path.join(DATASETS[dataset], model))
                df.dropna(inplace=True)
                after_dropna_shape = df.shape

                if sample_size:
                    df = df.sample(sample_size, random_state=2024)

                df['model'] = model.split('.')[0]
                df['length'] = df['student'].apply(lambda x: len(str(x).split()))
                
                logging.info(f"Average word length for model {model}: {df['length'].mean()}")
                
                if df.empty:
                    logging.warning(f"Empty DataFrame for {model} in {dataset}. Skipping.")
                    continue

                dfs.append(df)
            except FileNotFoundError:
                logging.warning(f"File {model} not found for {dataset}. Skipping.")
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file for {model} in {dataset}. Skipping.")
    
        if not dfs:
            logging.error(f"No data files were successfully loaded for {dataset}.")
            return
    return dfs