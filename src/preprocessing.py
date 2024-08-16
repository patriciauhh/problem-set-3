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
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')
    
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
    
    # Convert genres to uppercase 
    genre_list_upper = [genre.upper() for genre in genre_list]
    
    print("Columns in model_pred_df:", model_pred_df.columns)

    # Iterate through each row in model_pred_df to populate the dictionaries
    for _, row in model_pred_df.iterrows():
        actual_genres = row['actual genres']
        predicted_genre = row['predicted']
        
        # Convert actual genres string to list
        actual_genres_list = eval(actual_genres)

        for genre in genre_list_upper:
            true_col = f'{genre}_true'
            pred_col = f'{genre}_pred'

            if true_col not in model_pred_df.columns or pred_col not in model_pred_df.columns:
                print(f"Warning: Columns {true_col} or {pred_col} not found in DataFrame")
                continue  # Skip this genre if columns are missing
            
            # Update true count
            if genre in actual_genres_list:
                genre_true_counts[genre] += 1

            # Update true positive count
            if genre == predicted_genre and genre in actual_genres_list:
                genre_tp_counts[genre] += 1

            # Update false positive count
            if genre == predicted_genre and genre not in actual_genres_list:
                genre_fp_counts[genre] += 1
                
    return genre_list_upper, genre_true_counts, genre_tp_counts, genre_fp_counts
