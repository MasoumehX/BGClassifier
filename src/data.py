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
    data = data[~((data.x.isna()) | (data.y.isna())) | (data.point.isna()) | (data.name.isna()) | (data.frame.isna())
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


