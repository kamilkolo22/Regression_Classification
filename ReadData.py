import pandas as pd
import os


def read_regression_data(input_dir_path='input/Regression'):
    """Read data for regression problem"""
    data = {}
    for file in os.listdir(input_dir_path):
        data[file[:-4]] = pd.read_csv(
            input_dir_path + '/' + file, names=['col']).col.to_list()

    return pd.DataFrame.from_dict(data)


def read_classification_data(input_path='input/Classification/Telephones.csv'):
    return pd.read_csv(input_path)
