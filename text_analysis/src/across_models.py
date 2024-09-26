import pandas as pd
from utils.similarity import calculate_similarity
from config import MODELS

def analyze_across_models(dfs, col, truncate):
    results = []
    for i, df_i in enumerate(dfs):
        for j, df_j in enumerate(dfs[i+1:], start=i+1):
            min_rows = min(len(df_i), len(df_j))
            similarities = [calculate_similarity(df_i[col].iloc[k], df_j[col].iloc[k], truncate) for k in range(min_rows)]
            avg_similarities = pd.DataFrame(similarities).mean().round(3)
            results.append({
                'Model 1': MODELS[i].split('.')[0],
                'Model 2': MODELS[j].split('.')[0],
                **avg_similarities
            })
    return pd.DataFrame(results)

def analyze_student_vs_all_teachers(dfs, truncate):
    results = []
    for i, student_df in enumerate(dfs):
        for j, teacher_df in enumerate(dfs):
            if i != j:
                min_rows = min(len(student_df), len(teacher_df))
                similarities = [calculate_similarity(student_df['student'].iloc[k], teacher_df['teacher_summ'].iloc[k], truncate) for k in range(min_rows)]
                avg_similarities = pd.DataFrame(similarities).mean().round(3)
                results.append({
                    'Student Model': MODELS[i].split('.')[0],
                    'Teacher Model': MODELS[j].split('.')[0].lower().split('gpt2_')[1],
                    **avg_similarities
                })
    return pd.DataFrame(results)