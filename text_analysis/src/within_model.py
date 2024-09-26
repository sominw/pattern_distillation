import pandas as pd
from utils.similarity import calculate_similarity

def analyze_within_model(df, col1, col2, truncate):
    similarities = df.apply(lambda row: calculate_similarity(row[col1], row[col2], truncate), axis=1)
    return pd.DataFrame(list(similarities)).mean().round(3)