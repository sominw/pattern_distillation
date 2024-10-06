import sys
import os
import warnings
import logging
import pandas as pd
from datetime import datetime
import argparse

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)
sys.path.append(current_script_dir)

from config import DATASETS, MODELS, QA_DATASETS, QA_TEACHERS, QA_STUDENTS
from utils import setup_logging, map_qa_dataset, load_dfs
from src import (
    analyze_within_model,
    analyze_across_models,
    create_heatmap_plots,
    multiclass_lr_teacher_classification, 
    analyze_student_vs_all_teachers,
    create_student_teacher_heatmap
)

warnings.filterwarnings("ignore")

def run_analysis(dataset, truncate, same_ids, sample_size=None, qa_datasets=False):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    truncation_status = "truncated" if truncate else "original_len"
    output_dir = f"student_vs_teacher_exps/{dataset}/same_ids={same_ids}/{truncation_status}"

    if sample_size:
        output_dir += f"_n={sample_size}"
    os.makedirs(output_dir, exist_ok=True)
    

    dfs, original_dfs = load_dfs(dataset, truncate, same_ids, sample_size, qa_datasets)

    # # Within-Model Analysis (Student vs Teacher)
    # within_model_results = []

    # for i, df in enumerate(dfs):
    #     result = analyze_within_model(df, 'student', 'teacher_summ', truncate)
    #     within_model_results.append({'Teacher': MODELS[i].split('.')[0].split('gpt2_')[-1], **result})

    # pd.DataFrame(within_model_results).to_csv(os.path.join(output_dir, f'within_model_analysis_{truncation_status}.csv'), index=False)

    # # Between-Model Analysis (Student vs All Teacher)
    # student_vs_teachers_results = analyze_student_vs_all_teachers(dfs, truncate)
    # student_vs_teachers_results.to_csv(os.path.join(output_dir, f'student_vs_all_teachers_analysis_{truncation_status}.csv'), index=False)

    # # Heat map results
    # create_student_teacher_heatmap(student_vs_teachers_results, output_dir, truncation_status)

    # LR analysis; only run for sample n=50 for the students (not necessary for the others)
    if sample_size:
        accuracy, report, feature_importance = multiclass_lr_teacher_classification(dfs, original_dfs, truncate=truncate)
        # feature_importance.to_csv(os.path.join(output_dir, f'feature_importance_{truncation_status}.csv'), index=False)
        report = pd.DataFrame(report).transpose()
        report['model'] = report.index
        report.to_csv(os.path.join(output_dir, f'classification_report_{truncation_status}.csv'), index=False)

        logging.info(f"Results saved to {output_dir}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to analyze")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size to use")
    parser.add_argument('--qa_datasets', type=str, default='False')
    parser.add_argument('--same_ids', type=str, default='True')

    args = parser.parse_args()

    args.qa_datasets = args.qa_datasets.lower() == 'true'
    args.same_ids = args.same_ids.lower() == 'true'

    for truncate in [True, False]:
        logging.info(f"Running analysis for {args.dataset} dataset (truncate={truncate})...")
        run_analysis(args.dataset, truncate, args.same_ids, args.sample_size, args.qa_datasets)
        logging.info(f"Analysis for {args.dataset} (truncate={truncate}) completed.\n")

if __name__ == "__main__":
    main()