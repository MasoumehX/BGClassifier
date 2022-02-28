"Data preparation and pre-processing scripts"

import os
from utils import split_data, read_csv_file, write_csv_file, normalize
from plot import plot_power_law
from collections import Counter


def print_data_stats(df_data):
    print("========================== Columns ==========================")
    print()
    print(df_data.columns.tolist())
    print()
    print("data shape", df_data.shape)
    print()
    print("========================== Sample ===========================")
    print()
    print(df_data.head())
    print()
    print("========================== Linguistics ======================")
    print()
    total_types = df_data["words"].unique().shape[0]
    print("Nr. of types: ", total_types)
    name_groups = df_data.groupby(by="name")
    words_groups = [group.words.unique().shape[0] for name, group in name_groups]
    print("Nr. of tokens: ", sum(words_groups))
    print()
    print("========================== Word Frequencies ====================")
    print()
    groups = df_data.groupby(by=["name", "words"])
    words_list = [name[1] for name, group in groups]
    word_freq = Counter(words_list)
    min_freq = min(word_freq.values())
    max_freq = max(word_freq.values())

    lowest_freq = [k for k, v in word_freq.items() if v == min_freq]
    highest_freq = [k for k, v in word_freq.items() if v == max_freq]

    print("lowest frequency: ", min_freq)
    for w in lowest_freq:
        print(w)
    print("-" * 50)
    print("highest frequency: ", max_freq)
    for w in highest_freq:
        print(w)

    sorted_freqs = list(sorted(word_freq.values(), reverse=True))
    freq_of_freq = Counter(sorted_freqs)
    f = list(freq_of_freq.keys())
    ff = list(freq_of_freq.values())
    plot_power_law(f, ff, "Power Law for words in the Corpus", "./", "power_law_words", False, True)
    print()
    print("========================== Video ===============================")
    print()
    print("Nr. of videos: ", df_data.name.unique().shape[0])
    print()
    print("========================== Gestures Classes===============================")
    print()
    print("types per (semantic) class:")
    groups = df_data.groupby(by=["SemanticType"])
    for n, g in groups:
        print(" --- total types per ", n, " class:", g.words.unique().shape[0])
    print("total types: ", total_types)

    gesture_groups = {"demarcative": 0, "deictic": 0, "sequential": 0}
    dataset_groups = df_data.groupby(by=["SemanticType", "name"])
    for n, g in dataset_groups:
        if n[0] == "demarcative":
            gesture_groups["demarcative"] += g.words.unique().shape[0]
        if n[0] == "deictic":
            gesture_groups["deictic"] += g.words.unique().shape[0]
        if n[0] == "sequential":
            gesture_groups["sequential"] += g.words.unique().shape[0]
    print()
    print("gestures per (semantic) class:")
    print(" --- total gestures per `demarcative` class:", gesture_groups["demarcative"])
    print(" --- total gestures per `deictic` class:", gesture_groups["deictic"])
    print(" --- total gestures per `sequential` class:", gesture_groups["sequential"])
    print("total gestures: ", sum(gesture_groups.values()))


def general_pre_process(data, drop_cols=None, keep_point=None, keep_typePoint=None, keep_people=None):
    """ A helper function to clean the raw data"""
    # keep the pose (all poses)
    if keep_typePoint is not None:
        data = data[data.typePoint.isin(keep_typePoint)]
    # keep these points (hand gestures)
    if keep_point is not None:
        data = data[data.point.isin(keep_point)]
    # Drop unrelated columns (people, name, ...)
    if drop_cols is not None:
        data = data.drop(drop_cols, axis=1)
    # omit some people
    if keep_people is not None:
        data = data[data.people.isin(keep_people)]
    # Omit null or empty values
    data = data[~((data.x.isna()) | (data.y.isna()) | (data.point.isna()) | (data.name.isna()) | (data.frame.isna()) | (
        data.words.isna()))]
    # Omit zero values for x or y
    data = data[~((data.x == 0.0) | (data.y == 0.0))]
    # drop duplicated values
    data = data.drop_duplicates()
    # normalize the frame and get the time
    # TODO: I am not sure this convert frames to time!
    data["time"] = list(data.groupby("name").frame.apply(normalize))
    data = data[~data.time.isna()]
    return data


def special_pre_processing(df_data):
    """
        This script applies after merging the semantic class and the data.
        Look at the words that did not have a semantic class. Some of them
        may have a mis spelling.
    """
    df_data["words"] = df_data["words"].str.replace("subsequntly", "subsequently")
    df_data["words"] = df_data["words"].str.replace("ealier_than", "earlier_than")
    df_data["words"] = df_data["words"].str.replace("durtng_the_presidential", "during_the_presidential")
    df_data["words"] = df_data["words"].str.replace("during_her_entire", "during_the_entire")
    df_data["words"] = df_data["words"].str.replace("months_Ahead_of", "months_ahead_of")
    df_data["ww"] = df_data.words.apply(lambda x: 'distant_past' if x == 'distant_past_(1)' else x)
    df_data["words"] = df_data.ww
    return df_data


def data_cleaning(path_csv, path_txt, drop_cols=None, keep_point=None, keep_typepoint=None, keep_people=None,
                  special_prep=True):
    if keep_point is None:
        keep_point = []
    if drop_cols is None:
        drop_cols = []
    if keep_typepoint is None:
        keep_typepoint = []
    if keep_people is None:
        keep_people = []
    if not os.path.exists(path_csv):
        raise FileNotFoundError("File does not exist!")
    if path_csv.split(".")[-1] != "csv":
        raise ValueError("The file is not a .csv file! Try another function!")

    if not os.path.exists(path_txt):
        raise FileNotFoundError("File does not exist!")
    if path_txt.split(".")[-1] != "txt":
        raise ValueError("The file is not a .xlsx file! Try another function!")

    # reading the raw data
    print("started reading the .csv file ...")
    raw_data = read_csv_file(path_csv)

    # apply the general pre-processing
    print("general pre-processing started...")
    dff_clean = general_pre_process(raw_data, drop_cols=drop_cols, keep_point=keep_point, keep_typePoint=keep_typepoint,
                                    keep_people=keep_people)

    # apply special pre-processing
    if special_prep:
        print("started special pre-processing ...")
        dff_clean = special_pre_processing(dff_clean)

    # read and merge the semantic classes for all words
    df_semantic_classes = convert_txt_to_df(path_txt, separator="\t")
    print("-" * 60)
    print("Words that do not have semantic class: ")
    print(dff_clean[~dff_clean.words.isin(df_semantic_classes.Expression)].words.unique())
    data = dff_clean.merge(df_semantic_classes, left_on="words", right_on="Expression", how="inner")
    print("-" * 60)
    print("The new data base: ", data.shape)
    print("Columns to drop: ", drop_cols)
    print("Points to keep: ", keep_point)
    print("Poses to keep: ", keep_typepoint)
    print("people to keep: ", keep_people)
    print("-" * 60)

    # create numerical columns
    data = numerical_label(data, "words", "label")
    data = numerical_label(data, "SemanticType", "classes")
    data = numerical_label(data, "name", "fid")
    data["poi"] = data["point"].astype(int)

    # sorting based on the name (file name)
    data_sorted = data.sort_values(by=["fid", "time"])
    print("end of cleaning data.")
    return data_sorted


def create_data_for_training(data, with_pad=False, max_seq_len=None, features=None, model="base"):
    """ A helper function to create data for neural networks model"""

    groups = data.groupby(by=["fid", "words"])
    X_data = []
    y_data = []
    features_len = []
    for name, group in groups:
        f_stacks = np.stack(group[features].values)
        features_len.append(f_stacks.shape[0])
        X_data.append(f_stacks)
        class_name = group.classes.unique()
        if len(class_name) > 1:
            raise ValueError("A word can not have more than one class!")
        y_data.append(class_name[0])

    # Applying zero pad in two different ways (as a vector M x 1 x N or as a matrix M x D x F)
    if max_seq_len is not None:
        max_seq_ = max_seq_len
    else:
        max_seq_ = max(features_len)
    if with_pad and model == "nn":
        return padding_matrix(X_data, max_seq_len=max_seq_, feature_dim=len(features)), np.array(y_data), max_seq_
    elif with_pad and model == "base":
        return padding_vector(X_data), np.array(y_data), max_seq_
    else:
        return X_data, np.array(y_data), max_seq_


def main():
    # path to data on server
    txt_file_path = "/data/home/masoumeh/Data/classessem.txt"
    csv_file_path = "/data/home/agora/data/rawData/fullColectionCSV/fullColectionCSV|2022-01-19|03:42:12.csv"

    # cleaning the data
    keep_point = [2, 3, 4, 5, 6, 7]
    keep_typepoint = ['pose_keypoints']
    keep_people = [1]
    drop_cols = None
    feature_cols = ["x", "y", "poi", "time"]
    df_clean_data = data_cleaning(path_csv=csv_file_path, path_txt=txt_file_path, drop_cols=drop_cols,
                                  keep_point=keep_point, keep_typepoint=keep_typepoint,
                                  keep_people=keep_people)

    # split the data to train and test
    df_train, df_test = split_data(data=df_clean_data, train_ratio=80, sorting=True)
    df_data = pd.concat([df_train, df_test], sort=True)

    print("info of df_data (for train and test, replace `df_data` with `df_train` or `df_test`)")
    print_data_stats(df_data)

    # Writing the results in csv files
    where_to_write = "/data/home/masoumeh/Data/"
    write_csv_file(df_data, path=where_to_write + "nn/df_data.csv")
    write_csv_file(df_train, path=where_to_write + "nn/df_train.csv")
    write_csv_file(df_test, path=where_to_write + "nn/df_test.csv")
