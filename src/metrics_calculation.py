'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''
    # micro metrics
    
    micro_tp = sum(genre_tp_counts.values())
    micro_fp = sum(genre_fp_counts.values())
    micro_fn = sum(genre_true_counts[genre] - genre_tp_counts.get(genre, 0) for genre in genre_list)

    # compute micro precision, recall, and F1 score
    try:
        micro_precision = micro_tp / (micro_tp + micro_fp)
    except ZeroDivisionError:
        micro_precision = 0

    try:
        micro_recall = micro_tp / (micro_tp + micro_fn)
    except ZeroDivisionError:
        micro_recall = 0

    try:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    except ZeroDivisionError:
        micro_f1 = 0

    # initialize lists for macro metrics
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    # compute macro metrics for each genre
    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0)
        fp = genre_fp_counts.get(genre, 0)
        fn = genre_true_counts[genre] - tp

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
    
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
    
        # append metrics to their respective lists
        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    # return micro metrics as a tuple and macro metrics as lists
    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''
    
    # prepare lists to store the true and predicted values
    true_rows = []
    pred_rows = []
    
    # iterate over each row in model_pred_df
    for _, row in model_pred_df.iterrows():
        true_values = [row[f'{genre}_true'] for genre in genre_list]
        pred_values = [row[f'{genre}_pred'] for genre in genre_list]
        
        true_rows.append(true_values)
        pred_rows.append(pred_values)
    
    # Convert lists to matrices for sklearn
    true_matrix = np.array(true_rows)
    pred_matrix = np.array(pred_rows)
    
    # Flatten the matrices to compute metrics using sklearn
    true_flat = true_matrix.flatten()
    pred_flat = pred_matrix.flatten()
    
    # Calculate precision, recall, and F1 score using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average=None, labels=[1])
    
    macro_prec = precision.mean()
    macro_rec = recall.mean()
    macro_f1 = f1.mean()
    
    # Calculate micro metrics
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='micro', labels=[1])
    
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1


