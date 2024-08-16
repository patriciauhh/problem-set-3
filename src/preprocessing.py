'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returzns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('prediction_model.csv')
    genres_df = pd.read_csv('genres.csv')
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    genre_list = genres_df['genre'].unique().tolist()

    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    
    # iterate through each row in model_pred_df to populate the dictionaries
    for _, row in model_pred_df.iterrows():
        for genre in genre_list:
            true_col = f'{genre}_true'
            pred_col = f'{genre}_pred'
            
            # update true count
            genre_true_counts[genre] += row[true_col]
            
            # update true positive count
            if row[true_col] == 1 and row[pred_col] == 1:
                genre_tp_counts[genre] += 1
            
            # update false positive count
            if row[true_col] == 0 and row[pred_col] == 1:
                genre_fp_counts[genre] += 1
                
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts