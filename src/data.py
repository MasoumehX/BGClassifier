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


def create_data_for_nn(data, flat=False, with_zero_pad=False):
    """ A helper function to create a zero padded features for the data"""
    groups = data.groupby(by=["fid", "words"])
    X_data = []
    y_data = []
    for n, group in groups:
        features = np.stack(group[["x", "y", "poi"]].values)
        if flat:
            X_data.append(features.ravel())
        else:
            X_data.append(features)
        className = group.classes.unique()
        if len(className) > 1:
            raise ValueError("A word can not have more than one class!")
        y_data.append(className[0])
    if with_zero_pad:
        X_data = zero_pad(X_data)
    return X_data, y_data


def create_data_sets(path_csv, path_txt, drop_cols=None, keep_point=None, keep_typePoint=None, keep_people=None):

    if keep_point is None:
        drop_point = []
    if drop_cols is None:
        drop_cols = []
    if keep_typePoint is None:
        drop_pose = []
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
    dff_clean = general_pre_process(raw_data, drop_cols=drop_cols, keep_point=keep_points, keep_typePoint=keep_typepoint, keep_people=keep_people)

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
    print("Poses to keep: ", keep_typepoint)
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


if __name__ == '__main__':
    # local
    csv_file_path = "/home/masoumeh/Desktop/MasterThesis/Data/fullVideosClean.csv"
    txt_file_path = "/home/masoumeh/Desktop/MasterThesis/Data/classessem.txt"

    # server on ... lab
    # txt_file_path = "/data/home/masoumeh/Data/classessem.txt"
    # csv_file_path = "/data/home/agora/data/rawData/fullColectionCSV/fullColectionCSV|2022-01-19|03:42:12.csv"

    keep_points = [2, 3, 4, 5, 6, 7]
    keep_typepoint = ['pose_keypoints']
    keep_people = [1]
    drop_cols = None
    dataset, train, test = create_data_sets(path_csv=csv_file_path, path_txt=txt_file_path, drop_cols=drop_cols,
                                            keep_point=keep_points, keep_typePoint=keep_typepoint,
                                            keep_people=keep_people)

    print(dataset.words.unique())
    # count the words in each file
    dataset_groups = dataset.groupby("name")
    total_gestures = sum([g.words.unique().shape[0] for n, g in dataset_groups])
    print("total gestures in dataset: ", total_gestures)
    # count the words in each file
    train_groups = train.groupby("name")
    train_total = sum([g.words.unique().shape[0] for n, g in train_groups])
    print("total gestures in train: ", train_total)
    test_groups = test.groupby("name")
    test_total = sum([g.words.unique().shape[0] for n, g in test_groups])
    print("total gestures in test: ", test_total)
    print("total (semantic) classes: ", dataset.SemanticType.unique().shape[0])
    print("total (word) labels: ", dataset.words.unique().shape[0])

    dataset_groups = dataset.groupby(by=["SemanticType", "name"])
    gesture_groups={"demarcative":0, "deictic":0, "sequential":0}
    for n, g in dataset_groups:
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


    # ------------------------------------ Train -------------------------------------------------#

    gesture_groups = {"demarcative":0, "deictic":0, "sequential":0}
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


    # ------------------------------------ Test -------------------------------------------------#

    gesture_groups = {"demarcative":0, "deictic":0, "sequential":0}
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

    # TODO complete the metadata
    # metadata = pd.DataFrame(columns={"name", "n_rows", "n_gestures", "n_semantic_classes", "n_word_token",
    #                                  "n_word_type", "n_gesture_per_semantic", "n_gesture_per_word"})
    # db_name = ["small", "train_small", "test_small"]
    # n_rows = [dataset.shape[0], train.shape[0], test.shape[0]]
    # n_gestures = [total_gestures, train_total, test_total]
    #
    # metadata["n_rows"] = [dataset.shape[0], train[0]],

    # Writing the results in csv files
    # where_to_write = "/home/masoumeh/Desktop/MasterThesis/Data/"

    # server
    where_to_write = "/data/home/masoumeh/Data/"
    write_csv_file(dataset, path=where_to_write+"dataset_big_clean.csv")
    write_csv_file(train, path=where_to_write+"train_big_clean.csv")
    write_csv_file(test, path=where_to_write+"test_big_clean.csv")
    #
    # reading the csv files
    # read_csv_file(where_to_write+"dataset_big.csv")

    # path = "/home/masoumeh/Desktop/MasterThesis/Code/BodyGesturePatternDetection/docs/plots/"
    # plot_all_points_for_words(dataset, path=path)
    print("Done!")
