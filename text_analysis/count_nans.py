import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASETS = {
    'rotten_tomatoes': '/work/frink/shaib.c/pattern_distillation/inference/rotten_tomatoes/',
    'pubmed': '/work/frink/shaib.c/pattern_distillation/inference/pubmed/',
    'cnn': '/work/frink/shaib.c/pattern_distillation/inference/cnn/qa_outputs',
}

MODELS = [
    "gpt2_gemma2_9b.csv", 
    "gpt2_llama8b.csv", 
    "gpt2_mixtral.csv", 
    "gpt2_llama70b.csv", 
    "gpt2_mistral7b.csv",
]

def check_nans_in_file(file_path):
    try:
        df = pd.read_csv(file_path)
        total_rows = len(df)
        
        nan_counts = {}
        for column in df.columns:
            if df[column].dtype == 'object':  
                nan_count = df[column].isna().sum()
                if nan_count > 0:
                    nan_counts[column] = nan_count
        
        return total_rows, nan_counts
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return None, None

def main():
    for dataset, path in DATASETS.items():
        logging.info(f"Analyzing dataset: {dataset}")
        
        for model in MODELS:
            file_path = os.path.join(path, model)
            if os.path.exists(file_path):
                total_rows, nan_counts = check_nans_in_file(file_path)
                
                if total_rows is not None:
                    logging.info(f"  File: {model}")
                    logging.info(f"    Total rows: {total_rows}")
                    if nan_counts:
                        for column, count in nan_counts.items():
                            percentage = (count / total_rows) * 100
                            logging.info(f"    NaNs in '{column}': {count} ({percentage:.2f}%)")
                    else:
                        logging.info("    No NaNs found in text columns")
            else:
                logging.warning(f"  File not found: {file_path}")
        
        logging.info("")

if __name__ == "__main__":
    main()