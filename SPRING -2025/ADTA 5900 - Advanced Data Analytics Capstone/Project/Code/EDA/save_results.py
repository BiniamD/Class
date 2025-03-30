import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def setup_results_directory(results_dir='analysis_results'):
    """Create a directory to save results if it doesn't exist"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def save_plot(fig, filename, results_dir='analysis_results'):
    """Save a matplotlib figure to the results directory"""
    filepath = os.path.join(results_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved plot to {filepath}')

def save_dataframe(df, filename, results_dir='analysis_results'):
    """Save a pandas dataframe to CSV in the results directory"""
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)
    print(f'Saved dataframe to {filepath}')

def save_model_results(results_dict, filename, results_dir='analysis_results'):
    """Save model results to a JSON file"""
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f'Saved model results to {filepath}')

def save_training_history(history, filename, results_dir='analysis_results'):
    """Save training history to a JSON file"""
    filepath = os.path.join(results_dir, filename)
    history_dict = {}
    for key in history.history.keys():
        history_dict[key] = [float(val) for val in history.history[key]]
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f'Saved training history to {filepath}') 