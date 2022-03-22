import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import load
from sklearn.metrics import confusion_matrix
from plot import plot_confusion_matrix, plot_model_loss


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


def padding_vector(a, pad_value=-10, max_vec=None):
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
    df = df.apply(lambda row: np.pad(row['flatten'], pad_width=(0, max_vecdim - row['len']), constant_values=pad_value, mode='constant'),axis=1)
    return np.stack(df.values)


def padding_matrix(a, max_seq_len=None, pad_value=-10, feature_dim=4):
    """
    A helper function for padding an nd-array
    Parameter:
        data : a list of all features
    Returns:
        numpy array (N x max_vec x M) : N = len(a), max_vec = maximum length on a, M = a[0].shape[1]
    """
    Xpad = np.full((len(a), max_seq_len, feature_dim), fill_value=pad_value, dtype=float)
    print("max len: ", max_seq_len)
    for s, x in enumerate(a):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x
    return Xpad


def split_train_test(X, y, test_size=0.2, shuffle=True):
    """ returns numpy array"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=42, stratify=y)
    return X_train, y_train, X_test, y_test


def split_data(data, train_ratio=80, sorting=True):
    """
        Split the data to train and test
        :param sorting: sorting data at the end?
        :param train_ratio: the ratio between train and test. 80 means 80% for train and 20% for test.
        :param data
        :return pd.dataframe train and test

    """
    groups = data.groupby(['words'])
    test_ = []
    train_ = []
    for name, group in groups:
        files = group.name.unique()
        len_files = len(files)
        if len_files >= 3:
            ratio = int((len_files * train_ratio)/100)
            train_.extend(files[:ratio])
            test_.extend(files[ratio:])
    df_train = data[data.name.isin(train_)]
    df_test = data[data.name.isin(test_)]
    if sorting:
        df_train = df_train.sort_values(by=["fid", "time"])
        df_test = df_test.sort_values(by=["fid", "time"])
    return df_train, df_test


def get_files(path):
    if os.path.exists(path):
        return os.listdir(path)
    else:
        raise FileExistsError("Path does not exist!")


def read_multiple_csv(path, files):
    df_all = pd.DataFrame()
    for file in files:
        if os.path.exists(path+file):
            print("reading file ", file)
            df = pd.read_csv(path+file)
            df_all = df_all.append(df)
    return df_all


def load_dataset(set_name='head_bi_dem_norm_0.1_nn_CNN_5', where_to_read=None):
    xtrain = load(where_to_read + "/xtrain" + "_" + set_name + ".npy")
    ytrain = load(where_to_read + "/ytrain" + "_" + set_name + ".npy")
    xtest = load(where_to_read + "/xtest" + "_" + set_name + ".npy")
    ytest = load(where_to_read + "/ytest" + "_" + set_name + ".npy")
    max_seq_len = xtest[0].shape[0]
    return xtrain, ytrain, xtest, ytest, max_seq_len


def print_test_train_size(set_name='head_bi_dem_norm_0.1_nn_CNN_5', where_to_read=None):

    xtrain, ytrain, xtest, ytest, _ = load_dataset(set_name=set_name, where_to_read=where_to_read)
    print('train shape: ', xtrain.shape)
    print('test shape: ', xtest.shape)
    print('--------------------------------------------')
    print('Nr. Deictic (train): ', ytrain.tolist().count(0))
    print('Nr. Sequential (train): ', ytrain.tolist().count(1))
    print('Nr. Demarcative (train): ', ytrain.tolist().count(2))
    print('--------------------------------------------')
    print('Nr. Deictic (test): ', ytest.tolist().count(0))
    print('Nr. Sequential (test): ', ytest.tolist().count(1))
    print('Nr. Demarcative (test): ', ytest.tolist().count(2))


def compute_confusion_matrices(y_true,
                               y_pred,
                               path_to_save,
                               filename,
                               target_names,
                               with_plot=True):

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(cm)
    if with_plot:
        plot_confusion_matrix(cm,
                              normalize=False,
                              target_names=target_names,
                              save=True,
                              path=path_to_save+'confusion_matrix/',
                              filename='confusion_matrix_'+filename)

        plot_confusion_matrix(cm,
                              normalize=True,
                              target_names=target_names,
                              save=True,
                              path=path_to_save+'confusion_matrix/',
                              filename='confusion_matrix_normalized'+filename)
    print('Done!')