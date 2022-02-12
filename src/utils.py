import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_csv_file(path, separator=","):
    """
        A helper function to read a txt file. If does not read the entire file.
        :param path: a full path to the file.
        :param separator: A separator character. Usually is "," or ";".
        :return: A pandas.Dataframe object.

    """
    if not os.path.exists(path):
        raise FileNotFoundError("File does not exist!")
    if path.split(".")[-1] != "csv":
        raise ValueError("The file is not a .csv file! Try another function!")
    return pd.read_csv(path, sep=separator)


def read_excel_file(path):
    """
        A helper function to read a txt file. If does not read the entire file.
        :param path: a full path to the file.
        :return: A pandas.Dataframe object.

    """
    if not os.path.exists(path):
        raise FileNotFoundError("File does not exist!")
    if path.split(".")[-1] != "xlsx":
        raise ValueError("The file is not a .xlsx file! Try another function!")
    return pd.read_excel(path)


def read_txt_file(path):
    """
        A helper function to read a txt file. If does not read the entire file.
        :param path: a full path to the file.
        :return: Text Object to be read by read(), readline(), readlines()
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File does not exist!")
    if path.split(".")[-1] != "txt":
        raise ValueError("The file is not a .txt file! Try another function!")
    text = open(path, mode="r", encoding="utf-8")
    return text


def convert_txt_to_df(path, separator=" "):
    """
        A helper function to read a txt file and convert it to a pandas.Dataframe. The first line should be the header.
        :param path: a full path to the file.
        :param separator: A separator character. Usually is " " or "\t".
        :return: pandas.DataFrame
    """
    text = read_txt_file(path)
    header = next(text)
    header_line = [col.strip() for col in header.split(separator)]
    dts = []
    for line in text:
        content = line.strip().split(separator)
        if len(content) == 2:
            content = [c.strip() for c in content]
            dts.append(content)
    return pd.DataFrame(dts, columns=header_line)


def write_csv_file(df, path, separator=",", with_index=False):
    return df.to_csv(path, sep=separator, index=with_index)


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def numerical_label(df, col, new_col):
    """ A helper function to map the file names to a unique integer"""
    labels = df[col].unique()
    label_dic = dict()
    for index, lname in enumerate(labels):
        label_dic[lname] = index
    df[new_col] = df[col].map(label_dic)
    return df


def get_freq_power(data, freq_col):
    """ A helper function to compute the word frequency"""
    word_frequency = data[freq_col].value_counts()
    word_frequency = word_frequency.to_dict()
    keys = word_frequency.keys
    values = word_frequency.values
    return keys, values


def padding_vector(a, max_vec=None):
    """
    A function to zero padding
    Parameter:
        a : a list of all features
    Returns:
        numpy array
    """
    df = pd.DataFrame()
    df["features"] = a
    # flatten
    df["flatten"] = df["features"].apply(np.ravel)
    if max_vec is None:
        df['len'] = df['flatten'].apply(len)
        max_vecdim = df['len'].max()
        print("max len: ",max_vecdim)
    else:
        max_vecdim=max_vec
        print("max len: ",max_vecdim)
    df = df.apply(lambda row: np.pad(row['flatten'], pad_width=(0, max_vecdim - row['len']), constant_values=-10, mode='constant'),axis=1)
    return np.stack(df.values)


def padding_matrix(a, max_seq_len=None, pad_value=-10, feature_dim=4):
    """
    A helper function for padding an nd-array
    Parameter:
        data : a list of all features
    Returns:
        numpy array (N x max_vec x M) : N = len(a), max_vec = maximum length on a, M = a[0].shape[1]
    """
    Xpad = np.full((len(a), max_seq_len, feature_dim), fill_value=pad_value)
    for s, x in enumerate(a):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x
    return Xpad


def split_train_test(X, y, test_size=0.2, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=42)
    return X_train, y_train, X_test, y_test
