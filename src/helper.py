import settings # Read the config
import os

import pandas as pd
from termcolor import colored
from sklearn.model_selection import train_test_split

def console_general_data_info(data: pd.DataFrame) -> None:
    print(colored("\nGeneral data information:", "green"))
    print(colored("Example data:", "blue"))
    print(data.head())
    print(colored("Data types:", "blue"))
    print(data.dtypes)
    print(colored("Additional information:", "blue"))
    print(data.info, "\n")
    return

def split_data(data: pd.DataFrame):
    # Split test, validation and training data
    ## Split the DataFrame into features and target variable
    X = data.drop(columns=['f16', 'f17', 'f18', 'f31', 'f32'])
    y = data[('f16', 'pac泵1频率')]
    ## Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    ## Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    ## Print the shapes of each set
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test

def planned_obsolete(target):
    cnt = 0
    if target > cnt:
        cnt += 1
        yield
    else:
        raise Exception
    
def create_folder_if_not_exists(dir) -> bool:
    """
    Return bool, indicating if the folder is *newly* created
    """
    if not (isExist := os.path.exists(dir)):
        os.mkdir(dir)
    return isExist