import random

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

"""
def is_categorical(x):
    assert type(x) == pd.Series
    x = x.convert_dtypes()
    if is_string_dtype(x):
        return True
  
    elif set(x) == {0, 1}:
        return True
  
    elif x.dtype in [int, float, 'Int64', 'Float64']:
        return False
 
    else:
        return True
"""


def is_categorical(x: pd.Series, meta_data: dict, column_name: str) -> bool:
    """
    Checks if a column is categorical using the metadata JSON as the
    primary source of truth.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series")

    # Get the list of categorical names from the metadata
    categorical_feature_names = [f.get("name") for f in meta_data.get("categorical_features", [])]

    # Is the column name in our official list
    if column_name in categorical_feature_names:
        return True

    # Handle columns not in the metadata
    if column_name == "Result":
        return True

    # Fallback Heuristic
    x = x.convert_dtypes()
    if is_string_dtype(x):
        return True
    elif set(x) == {0, 1}:  # Catches binary
        return True
    elif x.dtype in [int, float, "Int64", "Float64"]:
        return False
    else:
        return True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def serialize(row):
    target_str = ""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx == 0:
            target_str += "If "
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join(
                [attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()]
            )
            target_str += ", "
        else:
            if len(attr_name.strip()) < 2:
                continue
            target_str += " Then "
            target_str += " is ".join(
                [attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()]
            )
            target_str += "."
    return target_str
