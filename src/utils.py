import os
import pandas as pd


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


def convert_txtfile_to_df(path, separator=" "):
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
