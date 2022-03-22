"Data preparation and pre-processing scripts"

from utils import *
from plot import plot_power_law
from collections import Counter
from numpy import save
from pathlib import Path


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
    data = data[~(data.isin([np.nan, np.inf, -np.inf]).any(1))]
    # Omit zero values for x or y
    data = data[~((data.x == 0.0) | (data.y == 0.0))]
    # drop duplicated values
    data = data.drop_duplicates()
    # normalize the frame and get the time
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


def data_cleaning(raw_data, path_txt, drop_cols=None, keep_point=None, keep_typepoint=None, keep_people=None,
                  special_prep=True):
    # read and merge the semantic classes for all words
    df_semantic_classes = convert_txt_to_df(path_txt, separator="\t")

    # apply the general pre-processing
    print("general pre-processing started...")
    dff_clean = general_pre_process(raw_data, drop_cols=drop_cols, keep_point=keep_point, keep_typePoint=keep_typepoint,
                                    keep_people=keep_people)

    # apply special pre-processing
    if special_prep:
        print("started special pre-processing ...")
        dff_clean = special_pre_processing(dff_clean)
    print("-" * 60)
    print("Words that do not have semantic class: ")
    print(dff_clean[~dff_clean.words.isin(df_semantic_classes.Expression)].words.unique())
    data = dff_clean.merge(df_semantic_classes, left_on="words", right_on="Expression", how="inner")
    data = data.drop(["ww", "Expression"], axis=1)
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
    data["poi"] = data["point"].astype(float)

    # sorting based on the name (file name)
    # data = data.sort_values(by=["fid", "time"])
    print("End of cleaning the data.")
    return data


def create_data_for_training(data, with_pad=False, class_col="classes", max_seq_len=None, features=None, pad_value=-10, model="base"):
    """ A helper function to create data for neural networks model"""

    groups = data.groupby(by=["fid", "words"])
    X_data = []
    y_data = []
    features_len = []
    for name, group in groups:
        f_stacks = np.stack(group[features].values)
        features_len.append(f_stacks.shape[0])
        X_data.append(f_stacks)
        class_name = group[class_col].unique()
        if len(class_name) > 1:
            raise ValueError("A word can not have more than one class!")
        y_data.append(class_name[0])

    # Applying zero pad in two different ways (as a vector M x 1 x N or as a matrix M x D x F)
    if max_seq_len is not None:
        max_seq_ = max_seq_len
    else:
        max_seq_ = max(features_len)
    if with_pad and model.startswith("nn"):
        return padding_matrix(X_data, max_seq_len=max_seq_, feature_dim=len(features), pad_value=pad_value), np.array(y_data), max_seq_
    elif with_pad and model.startswith("base"):
        return padding_vector(X_data, pad_value=pad_value), np.array(y_data), max_seq_
    else:
        return X_data, np.array(y_data), max_seq_


def create_train_test(df_data,
                      setname='head',
                      test_size=0.3,
                      with_pad=True,
                      pad_value=-10,
                      feature_cols=None,
                      features_type='norm',
                      class_col="multi",
                      keep_points=None,
                      model="nn_cnn",
                      root="./"):
    if keep_points is not None:
        df_data = df_data[df_data.point.isin(keep_points)]

    if feature_cols is None or len(feature_cols) < 1:
        raise ValueError("The features can not be empty!")

    print("Create data for ", model)
    xdata, ydata, max_seq_len = create_data_for_training(df_data,
                                                         with_pad=with_pad,
                                                         class_col=class_col,
                                                         pad_value=pad_value,
                                                         features=feature_cols,
                                                         model=model)

    print("split the data to train and test!")
    xtrain, ytrain, xtest, ytest = split_train_test(xdata, ydata, test_size=test_size, shuffle=True)

    filename = setname + '_' + class_col + '_' + features_type + '_' + str(test_size) + '_' + model
    foldername = '/'.join([setname, model, str(test_size), class_col, features_type])
    path_to_save = os.path.join(root, foldername)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    print("Saving train and test for  ", filename)
    save(path_to_save + "/xtrain" + "_" + filename + ".npy", xtrain)
    save(path_to_save + "/ytrain" + "_" + filename + ".npy", ytrain)
    save(path_to_save + "/xtest" + "_" + filename + ".npy", xtest)
    save(path_to_save + "/ytest" + "_" + filename + ".npy", ytest)
    return xtrain, ytrain, xtest, ytest


def create_raw_data():
    csv_file_path = "/mnt/shared/people/masoumeh/MA/data/data.csv"
    # reading the raw data
    print("start reading the .csv file ...")
    return read_csv_file(csv_file_path)


def create_clean_data(df_raw_data, save_data=True):
    # path to data on server
    path_txt = "/mnt/shared/people/masoumeh/MA/data/classessem.txt"
    # cleaning the data
    keep_point = None
    keep_typepoint = ['pose_keypoints']
    keep_people = [1]
    drop_cols = ["u", "typePoint", "x2", "y2"]
    df_clean_data = data_cleaning(df_raw_data, path_txt=path_txt, drop_cols=drop_cols,
                                  keep_point=keep_point, keep_typepoint=keep_typepoint,
                                  keep_people=keep_people)
    print("info of df_data (for train and test, replace `df_data` with `df_train` or `df_test`)")
    print_data_stats(df_clean_data)

    if save_data:
        # Writing the results in csv files
        where_to_write = "/mnt/shared/people/masoumeh/MA/data/"
        write_csv_file(df_clean_data, path=where_to_write + "df_clean_data.csv")
    return df_clean_data


if __name__ == '__main__':
    df_raw = create_raw_data()
    create_clean_data(df_raw, save_data=False)




