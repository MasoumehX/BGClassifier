"Data preparation and pre-processing scripts"

import os
from collections import Counter
from utils import read_csv_file
from plot import plot_freq_power


def print_data_stats(df_data):
    print("raw data shape: ", df_data.shape)
    df_data_nan = df_data[(df_data.x.isna()) & (df_data.y.isna())]
    print(df_data_nan)
    words_raw_q = df_data.words.unique()
    print("raw word types: ", len(words_raw_q))
    groups = df_data.groupby(by="words")
    words_groups = [group.shape[0] for name, group in groups]
    print("lowest freq: ", min(words_groups))
    print("highest freq: ", max(words_groups))
    print("==============================================================")
    df_data_clean = df_data[(~df_data.x.isna()) & (~df_data.y.isna())]
    print("clean data shape: ", df_data_clean.shape)
    print(df_data_clean)
    words_q = df_data_clean.words.unique()
    print("cleaned word types: ", len(words_q))
    groups = df_data_clean.groupby(by="words")
    words_groups = [group.shape[0] for name, group in groups]
    print("lowest freq: ", min(words_groups))
    print("highest freq: ", max(words_groups))
    print("Unique words: ", words_q)


def pre_process(data):
    df_data_clean = data[(~data.x.isna()) & (~data.y.isna())]
    print("clean data shape: ", df_data_clean.shape)
    return df_data_clean


def get_freq_power(df_data, freq_col):
    word_frequency = df_data[freq_col].value_counts()
    types = word_frequency.keys().tolist()
    counts = word_frequency.values
    return types, counts


if __name__ == '__main__':
    path = "/home/masoumeh/Desktop/MasterThesis/Data/"
    filename = "fullVideosClean_with_SemanticType.csv"
    df = read_csv_file(os.path.join(path, filename))
    df_clean = pre_process(df)
    # print(df_clean.columns)
    # print(df_clean.words.unique())
    print(df_clean.SemanticType.unique())
    deictic = df_clean[df_clean.SemanticType == "deictic"]
    print(deictic.words.unique())
    deictic_past = deictic[deictic.words.str.contains('past')]
    all_past = df_clean[df_clean["words"].str.contains('past', na=False)]
    # print(deictic_past.words.unique())
    # print(all_past.words.unique())


    # print_data_stats(df)
    # keys, values = get_freq_power(df, "words")
    # keys, values = get_freq_power(df_clean, "words")
    # print(len(f), len(ff))
    # print(f, ff)
    # plot_freq_power(keys, values, save=True, title="raw")
    # df_sample = df_clean[df_clean.words == "from_beginning_to_end"]
    # fname = df_sample["name"].iloc[0]
    # print(fname)
    # df_sample_from_beginning_to_end = df_sample[df_sample["name"] == fname]
    # print(df_sample_from_beginning_to_end)
    # print(df_sample_from_beginning_to_end.point)