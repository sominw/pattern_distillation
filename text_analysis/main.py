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
    analyze_student_vs_all_teachers,
    pairwise_lr_analysis,
    student_teacher_pairwise_lr,
    analyze_data,
    create_heatmap_plots
)

warnings.filterwarnings("ignore")

def run_analysis(dataset, truncate, same_ids, sample_size=None, qa_datasets=False):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    truncation_status = "truncated" if truncate else "original_len"
    output_dir = f"same_ids={same_ids}/{dataset}_{truncation_status}"

    if sample_size:
        output_dir += f"_n={sample_size}"
    os.makedirs(output_dir, exist_ok=True)
    

    dfs = load_dfs(dataset, truncate, same_ids, sample_size, qa_datasets)

    # Within-Model Analysis (Student vs its own Teacher)
    within_model_results = []
    for i, df in enumerate(dfs):
        result = analyze_within_model(df, 'student', 'teacher_summ', truncate)
        within_model_results.append({'Model': MODELS[i].split('.')[0], **result})
    pd.DataFrame(within_model_results).to_csv(os.path.join(output_dir, f'within_model_analysis_{truncation_status}.csv'), index=False)
    
    # Between-Model Analysis (Student vs Student)
    across_models_results = analyze_across_models(dfs, 'student', truncate)
    across_models_results.to_csv(os.path.join(output_dir, f'across_models_analysis_{truncation_status}.csv'), index=False)
    
    # Create heatmap plots for across-models analysis
    create_heatmap_plots(across_models_results, output_dir, truncation_status)
    
    # Between-Model Analysis (Student vs Different Teacher)
    student_vs_teachers_results = analyze_student_vs_all_teachers(dfs, truncate)
    student_vs_teachers_results.to_csv(os.path.join(output_dir, f'student_vs_teachers_analysis_{truncation_status}.csv'), index=False)

    # Between-Model LR (Student vs Teacher)
    student_teacher_lr = student_teacher_pairwise_lr(dfs, truncate)
    student_teacher_lr.to_csv(os.path.join(output_dir, f'student_teacher_pairwise_lr_analysis_{truncation_status}.csv'), index=False)
    
    # Student Pairwise LR
    pairwise_student_lr = pairwise_lr_analysis(dfs, 'student', 'model', 'pairwise_student_summaries', truncate)
    pairwise_student_lr.to_csv(os.path.join(output_dir, f'pairwise_student_lr_analysis_{truncation_status}.csv'), index=False)
    
    # All Models LR
    model_classification = analyze_data('student', 'model', 'models', dfs, truncate)
    teacher_classification = analyze_data('teacher_summ', 'teacher', 'teachers', dfs, truncate)
    
    if model_classification and teacher_classification:
        all_models_lr = pd.DataFrame({
            'Analysis': ['Model Classification', 'Teacher Classification'],
            'Accuracy': [model_classification['accuracy'], teacher_classification['accuracy']]
        })
        all_models_lr.to_csv(os.path.join(output_dir, f'all_models_lr_analysis_{truncation_status}.csv'), index=False)
    else:
        logging.warning("All Models LR analysis could not be performed due to data issues.")
    
    logging.info(f"Results saved to {output_dir}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to analyze")
    # parser.add_argument("--truncate", type=bool, default=False, help="Whether to truncate text")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size to use")
    parser.add_argument('--qa_datasets', type=str, default='False')
    parser.add_argument('--same_ids', type=str, default='True')

    args = parser.parse_args()

    args.qa_datasets = args.qa_datasets.lower() == 'true'
    args.same_ids = args.same_ids.lower() == 'true'

    # for dataset in DATASETS:
    for truncate in [True, False]:
        logging.info(f"Running analysis for {args.dataset} dataset (truncate={truncate})...")
        # run_analysis(dataset, truncate, sample_size=None, qa_datasets=False)
        run_analysis(args.dataset, truncate, args.same_ids, args.sample_size, args.qa_datasets)
        logging.info(f"Analysis for {args.dataset} (truncate={truncate}) completed.\n")

if __name__ == "__main__":
    main()