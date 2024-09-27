import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

def create_student_teacher_heatmap(df, output_dir, truncation_status):
    sns.set_style("white", {'axes.grid': False})
    
    student_models = sorted(df['Student Model'].unique())
    teacher_models = sorted(df['Teacher Model'].unique())
    
    metrics = ['cosine_similarity', 'rouge_1_f', 'rouge_2_f', 'rouge_l_f']
    
    for metric in metrics:
        plt.figure(figsize=(12, 10))
        
        heatmap_data = pd.DataFrame(index=student_models, columns=teacher_models, dtype=float)
        
        for _, row in df.iterrows():
            student = row['Student Model']
            teacher = row['Teacher Model']
            value = row[metric]
            try:
                heatmap_data.loc[student, teacher] = float(value)
            except (ValueError, TypeError):
                heatmap_data.loc[student, teacher] = np.nan
        
        mask = np.zeros_like(heatmap_data, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        cmap = sns.light_palette("navy", as_cmap=True)
        
        sns.heatmap(heatmap_data, annot=True, cmap=cmap, fmt='.2f', linewidths=0.5,
                    cbar=False, linecolor='white', annot_kws={"size": 10, "weight": "bold"},
                    mask=mask)
        
        metric_name = metric.replace('_', ' ').title()
        plt.title(f"{metric_name} - Student vs Teacher Models ({truncation_status})", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Teacher Model', fontsize=12, fontweight='bold')
        plt.ylabel('Student Model', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_filename = os.path.join(output_dir, f"student_teacher_{metric.lower()}_{truncation_status}.pdf")
        plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Student-Teacher heatmap plots saved in the folder '{output_dir}'")


def create_heatmap_plots(df, output_dir, truncation_status):
    sns.set_style("white", {'axes.grid': False})
    
    df['Model 1'] = df['Model 1'].str.replace('gpt2_', '')
    df['Model 2'] = df['Model 2'].str.replace('gpt2_', '')
    
    models = sorted(set(df['Model 1'].unique()) | set(df['Model 2'].unique()))
    
    metrics = ['cosine_similarity', 'rouge_1_f', 'rouge_2_f', 'rouge_l_f']
    metric_dfs = {metric: pd.DataFrame(index=models, columns=models, dtype=float) for metric in metrics}
    
    for _, row in df.iterrows():
        for metric in metrics:
            metric_dfs[metric].loc[row['Model 2'], row['Model 1']] = row[metric]
            metric_dfs[metric].loc[row['Model 1'], row['Model 2']] = row[metric]
    
    for metric in metrics:
        np.fill_diagonal(metric_dfs[metric].values, 1.0)
    
    cmap = sns.light_palette("navy", as_cmap=True)
    
    for metric, data in metric_dfs.items():
        plt.figure(figsize=(8, 6))
        
        mask = np.triu(np.ones_like(data, dtype=bool), k=1)
        
        data_masked = data.mask(np.eye(data.shape[0], dtype=bool), other=np.nan)
     
        vmin = data_masked.min().min()
        vmax = data_masked.max().max()
        
        sns.heatmap(data, annot=True, cmap=cmap, fmt='.3f', square=True, linewidths=0.5,
                    vmin=vmin, vmax=vmax, annot_kws={"size": 10, "weight": "bold"}, 
                    mask=mask, cbar=False, linecolor='white', linewidth=0.5)
    
        for i in range(data.shape[0]):
            plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='white', edgecolor='white'))
    
        metric_name = metric.replace('_', ' ').title()
        plt.title(f"{metric_name.replace('F', '')} ({truncation_status})", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
    
        plot_filename = os.path.join(output_dir, f"{metric_name.replace(' ', '_')}_{truncation_status}.pdf")
        plt.savefig(plot_filename, format='pdf', dpi=300)
        plt.close()
    
    logging.info(f"Heatmap plots saved in the folder '{output_dir}'")