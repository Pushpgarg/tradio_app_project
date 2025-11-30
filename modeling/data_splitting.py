from typing import Literal
from sklearn.model_selection import train_test_split

def split_data(data, target_col = 'target' ,test_size=0.3, random_state=42, stratify_col=None , data_split_type: Literal["random","sequential"]="random"):
    if data_split_type == "random":
        x_train , x_test , y_train, y_test = train_test_split(data.drop(columns=[target_col]), data[target_col], test_size=test_size, random_state=random_state, stratify=data[stratify_col] if stratify_col else None)

    elif data_split_type == "sequential":
        n_rows = data.shape[0]
        train_size = int(n_rows * (1 - test_size))

        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        x_train , y_train = train_data.drop(columns=[target_col]), train_data[target_col]
        x_test , y_test = test_data.drop(columns=[target_col]), test_data[target_col]

    else:
        raise ValueError("data_split_type must be either 'random' or 'sequential'")
    
    return x_train, x_test, y_train, y_test