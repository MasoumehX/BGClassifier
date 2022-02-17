"Data preparation and pre-processing scripts"

from utils import *


def print_data_stats(df_data):
    print("========================== Columns ==========================")
    print()
    print(df_data.columns.tolist())
    print()
    print(df_data.shape)
    print()
    print("========================== Sample ===========================")
    print()
    print(df_data.head())
    print()
    print("========================== Linguistics ======================")
    print()
    words_q = df_data["words"].unique()
    print("Nr. of types: ", len(words_q))
    print("Nr. of tokens: ", df_data.shape[0])
    print()
    print("========================== Stats ============================")
    print()
    groups = df_data.groupby(by="words")
    words_groups = [group.file.unique().shape[0] for name, group in groups]
    print("Word Frequency:")
    print(" ---- lowest frequency: ", min(words_groups))
    print(" ---- highest frequency: ", max(words_groups))


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
    data = data[~((data.x.isna()) | (data.y.isna()) | (data.point.isna()) | (data.name.isna()) | (data.frame.isna()) | (data.words.isna()))]
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


def create_df_dataset_train_test(path_csv, path_txt, drop_cols=None, keep_point=None, keep_typePoint=None, keep_people=None):

    if keep_point is None:
        keep_point = []
    if drop_cols is None:
        drop_cols = []
    if keep_typePoint is None:
        keep_typePoint = []
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
    raw_data = read_csv_file(path_csv)

    # apply the general pre-processing
    dff_clean = general_pre_process(raw_data, drop_cols=drop_cols, keep_point=keep_point, keep_typePoint=keep_typePoint, keep_people=keep_people)

    # apply special pre-processing
    # dff_clean = special_pre_processing(dff_clean)

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
    print("Poses to keep: ", keep_typePoint)
    print("people to keep: ", keep_people)
    print("-" * 60)

    # create numerical columns
    data = numerical_label(data, "words", "label")
    data = numerical_label(data, "SemanticType", "classes")
    data = numerical_label(data, "name", "fid")
    data["poi"] = data["point"].astype(int)

    # Split the data to test, train and maybe val
    groups = data.groupby(['words'])
    test_ = []
    # val_ = []
    train_ = []
    for name, group in groups:
        files = group.name.unique()
        len_files = len(files)
        if len_files >= 3:
            percentage_80 = int((len_files * 80)/100)
            # percentage_20 = int((len_files * 15)/100)
            # print(len_files, percentage_80)
            train_.extend(files[:percentage_80])
            # val_.extend(files[percentage_70:percentage_70+percentage_15])
            test_.extend(files[percentage_80:])
    df_train = data[data.name.isin(train_)]
    # df_val = df[df.name.isin(val_)]
    df_test = data[data.name.isin(test_)]

    # sorting based on the name (file name)
    data_sorted = data.sort_values(by=["fid", "time"])
    df_train_sorted = df_train.sort_values(by=["fid", "time"])
    df_test_sorted = df_test.sort_values(by=["fid", "time"])

    return data_sorted, df_train_sorted, df_test_sorted


def create_data_for_training(data, with_pad=False, features=None, model="base"):
    """ A helper function to create data for neural networks model"""
    groups = data.groupby(by=["fid", "words"])
    X_data = []
    y_data = []
    features_len = []
    if features is None:
        features = []
    for name, group in groups:
        features = np.stack(group[features].values)
        features_len.append(features.shape[0])
        X_data.append(features)
        class_name = group.classes.unique()
        if len(class_name) > 1:
            raise ValueError("A word can not have more than one class!")
        y_data.append(class_name[0])

    # Applying zero pad in two different ways (as a vector M x 1 x N or as a matrix M x D x F)
    if with_pad and model == "nn":
        return padding_matrix(X_data, max_seq_len=max(features_len)), np.array(y_data)
    elif with_pad and model == "base":
        return padding_vector(X_data), np.array(y_data)
    else:
        return X_data, np.array(y_data)


def print_dataset_info(dataset, train, test):

    # ------------------------------------ dataset -------------------------------------------------#
    gesture_groups = {"demarcative": 0, "deictic": 0, "sequential": 0}
    train_groups = train.groupby(by=["SemanticType", "name"])
    for n, g in train_groups:
        if n[0] == "demarcative":
            gesture_groups["demarcative"] += g.words.unique().shape[0]
        if n[0] == "deictic":
            gesture_groups["deictic"] += g.words.unique().shape[0]
        if n[0] == "sequential":
            gesture_groups["sequential"] += g.words.unique().shape[0]

    print("total words: ", dataset.words.unique().shape[0])
    print("total gestures per (semantic) class:")
    print(" --- total gestures per `demarcative` class:", gesture_groups["demarcative"])
    print(" --- total gestures per `deictic` class:", gesture_groups["deictic"])
    print(" --- total gestures per `sequential` class:", gesture_groups["sequential"])
    print("total gestures: ", sum(gesture_groups.values()))

    # ------------------------------------ Train -------------------------------------------------#

    gesture_groups = {"demarcative": 0, "deictic": 0, "sequential": 0}
    train_groups = train.groupby(by=["SemanticType", "name"])
    for n, g in train_groups:
        if n[0] == "demarcative":
            gesture_groups["demarcative"] += g.words.unique().shape[0]
        if n[0] == "deictic":
            gesture_groups["deictic"] += g.words.unique().shape[0]
        if n[0] == "sequential":
            gesture_groups["sequential"] += g.words.unique().shape[0]

    print("total gestures per (semantic) class:")
    print(" --- total gestures per `demarcative` class:", gesture_groups["demarcative"])
    print(" --- total gestures per `deictic` class:", gesture_groups["deictic"])
    print(" --- total gestures per `sequential` class:", gesture_groups["sequential"])
    print("total gestures: ", sum(gesture_groups.values()))

    # ------------------------------------ Test -------------------------------------------------#

    gesture_groups = {"demarcative": 0, "deictic": 0, "sequential": 0}
    test_groups = test.groupby(by=["SemanticType", "name"])
    for n, g in test_groups:
        if n[0] == "demarcative":
            gesture_groups["demarcative"] += g.words.unique().shape[0]
        if n[0] == "deictic":
            gesture_groups["deictic"] += g.words.unique().shape[0]
        if n[0] == "sequential":
            gesture_groups["sequential"] += g.words.unique().shape[0]

    print("total gestures per (semantic) class:")
    print(" --- total gestures per `demarcative` class:", gesture_groups["demarcative"])
    print(" --- total gestures per `deictic` class:", gesture_groups["deictic"])
    print(" --- total gestures per `sequential` class:", gesture_groups["sequential"])
    print("total gestures: ", sum(gesture_groups.values()))