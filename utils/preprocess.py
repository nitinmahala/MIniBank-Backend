import numpy as np
import pandas as pd

def preprocess_input(data_dict):
    """
    Prepares raw transaction input for model prediction.

    Parameters
    ----------
    data_dict : dict
        Dictionary of feature_name: value pairs from the frontend.

    Returns
    -------
    DataFrame
        A single-row pandas DataFrame with the same structure as model input.
    """
    # Convert dictionary to DataFrame (1 row)
    df = pd.DataFrame([data_dict])
    return df
