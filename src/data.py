"Data preparation and pre-processing scripts"

from utils import *
from plot import plot_scatter


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


def pre_process(data, drop_cols=None, drop_point=None, drop_pose=None):
    """ A helper function to clean the raw data"""

    # rename columns
    if drop_pose is not None:
        data = data[~data.typePoint.isin(drop_pose)]

    # Omit unrelated points (Head, lower body, center)
    if drop_point is not None:
        data = data[~data.point.isin(drop_point)]

    # Drop unrelated columns
    if drop_cols is not None:
        data = data.drop(drop_cols, axis=1)

    # Omit null or empty values
    data = data[~((data.nx.isna()) | (data.ny.isna())) | (data.point.isna()) | (data.name.isna()) | (data.frame.isna())
                | (data.SemanticType.isna()) | (data.words.isna())]

    # drop duplicated values
    data = data.drop_duplicates()

    # filter words less than frequency 5
    # groups = data.groupby(['words'])
    # data = groups.filter(lambda x: x.file.unique().shape[0] >= 5)

    # create numerical columns
    data = numerical_label(data, "words", "label")
    data = numerical_label(data, "SemanticType", "classes")
    data = numerical_label(data, "name", "fid")

    # normalize the frame and get the time
    # TODO: I am not sure this convert frames to time!
    data["time"] = list(data.groupby("name").frame.apply(normalize))

    # convert the columns to int
    data["poi"] = data["point"].astype(int)
    data = data[~data.time.isna()]
    return data


def create_data_for_nn(path):
    """ A helper function to create a zero padded features for the data"""
    data = read_csv_file(path)
    groups = data.groupby(by=["fid", "words"])

    X_data = []
    y_data = []
    for n, group in groups:
        features = np.stack(group[["nx", "ny", "poi"]].values)
        X_data.append(features.ravel())
        y_data.append(group.classes.unique()[0])
    X_data = zero_pad(X_data)
    return X_data, y_data


def create_dataset(path_csv, path_txt, drop_cols=None, drop_point=None, drop_pose=None):

    if drop_point is None:
        drop_point = []
    if drop_cols is None:
        drop_cols = []
    if drop_pose is None:
        drop_pose = []
    if not os.path.exists(path_csv):
        raise FileNotFoundError("File does not exist!")
    if path_csv.split(".")[-1] != "csv":
        raise ValueError("The file is not a .csv file! Try another function!")

    if not os.path.exists(path_txt):
        raise FileNotFoundError("File does not exist!")
    if path_txt.split(".")[-1] != "txt":
        raise ValueError("The file is not a .xlsx file! Try another function!")

    # reading the two files
    df_data = read_csv_file(path_csv)
    print(df_data.typePoint.unique())
    df_semantic_classes = convert_txtfile_to_df(path_txt, separator="\t")
    print("-" * 60)
    print("Words that do not have semantic class: ")
    print(df_data[~df_data.words.isin(df_semantic_classes.Expression)].words.unique())
    data = df_data.merge(df_semantic_classes, left_on="words", right_on="Expression", how="inner")
    print("-" * 60)
    print("The new data base: ", data.shape)
    print("Columns to drop: ", drop_cols)
    print("Points to drop: ", drop_point)
    print("Poses to drop: ", drop_pose)
    print("-" * 60)
    dff = pre_process(data, drop_cols=drop_cols, drop_point=drop_point, drop_pose=drop_pose)
    # Split the data to test, train and maybe val

    groups = dff.groupby(['words'])
    test_ = []
    # val_ = []
    train_ = []
    for name, group in groups:
        files = group.name.unique()
        len_files = len(files)
        percentage_80 = int((len_files * 80)/100)
        # percentage_20 = int((len_files * 15)/100)
        # print(len_files, percentage_80)
        train_.extend(files[:percentage_80])
        # val_.extend(files[percentage_70:percentage_70+percentage_15])
        test_.extend(files[percentage_80:])

    df_train = dff[dff.name.isin(train_)]
    # df_val = df[df.name.isin(val_)]
    df_test = dff[dff.name.isin(test_)]

    # sorting based on the name (file name)
    dff = dff.sort_values(by=["fid", "time"])
    df_train = df_train.sort_values(by=["fid", "time"])
    df_test = df_test.sort_values(by=["fid", "time"])

    return dff, df_train, df_test


if __name__ == '__main__':
    # local
    csv_file_path = "/home/masoumeh/Desktop/MasterThesis/Data/fullVideosClean.csv"
    txt_file_path = "/home/masoumeh/Desktop/MasterThesis/Data/classessem.txt"

    # server on ... lab
    # txt_file_path = "/data/home/masoumeh/Data/classessem.txt"
    # csv_file_path = "/data/home/agora/data/rawData/fullColectionCSV/fullColectionCSV|2022-01-19|03:42:12.csv"

    non_related_points = [8, 9, 12, 10, 13, 11, 24, 23, 22, 21, 14, 19, 20, 15, 16, 17, 18, 0, 1]
    dataset, train, test = create_dataset(path_csv=csv_file_path, path_txt=txt_file_path, drop_point=non_related_points)

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

    # TODO complete the metadata
    # metadata = pd.DataFrame(columns={"name", "n_rows", "n_gestures", "n_semantic_classes", "n_word_token",
    #                                  "n_word_type", "n_gesture_per_semantic", "n_gesture_per_word"})
    # db_name = ["small", "train_small", "test_small"]
    # n_rows = [dataset.shape[0], train.shape[0], test.shape[0]]
    # n_gestures = [total_gestures, train_total, test_total]
    #
    # metadata["n_rows"] = [dataset.shape[0], train[0]],

    # Writing the results in csv files
    where_to_write = "/home/masoumeh/Desktop/MasterThesis/Data/"
    # write_csv_file(dataset, path=where_to_write+"dataset_small_small.csv")
    # write_csv_file(train, path=where_to_write+"train_small_small.csv")
    # write_csv_file(test, path=where_to_write+"test_small_small.csv")
    #
    # reading the csv files
    # read_csv_file(where_to_write+"dataset_big.csv")

    # path = "/home/masoumeh/Desktop/MasterThesis/Code/BodyGesturePatternDetection/docs/plots/"
    # plot_all_points_for_words(dataset, path=path)
    print("Done!")
