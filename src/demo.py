from data import *

local_root = "/home/masoumeh/Desktop/MasterThesis/Data/"
csv_file_path = "fullVideosClean.csv"
txt_file_path = "classessem.txt"
server_root = "/data/home/masoumeh/Data/"
server_csv_file_path = "/data/home/agora/data/rawData/fullColectionCSV/fullColectionCSV|2022-01-19|03:42:12.csv"

non_related_points = [8, 9, 12, 10, 13, 11, 24, 23, 22, 21, 14, 19, 20, 15, 16, 17, 18, 0, 1]
dataset, train, test = create_dataset(path_csv=server_csv_file_path, path_txt=server_root+txt_file_path, drop_point=non_related_points)


cols = ["x", "y", "point", "time", "frame", "tid", "classes", "label"]
train = train[cols]
print(train.label.unique().shape)
print(test.label.unique().shape)
# print(val.words.unique().shape)

print(train[train.SemanticType == "demarcative"].file.unique().shape)
print(train[train.SemanticType == "deictic"].file.unique().shape)
print(train[train.SemanticType == "sequential"].file.unique().shape)

print(test[test.SemanticType == "demarcative"].file.unique().shape)
print(test[test.SemanticType == "deictic"].file.unique().shape)
print(test[test.SemanticType == "sequential"].file.unique().shape)

write_csv_file(dataset, path=server_root+"dataset_big.csv")
write_csv_file(train, path=server_root+"train_big.csv")
write_csv_file(test, path=server_root+"test_big.csv")
# write_csv_file(val, path=root+"val_3.csv")


# dd = read_csv_file(root+"val.csv")
# ddd = val[val.tid==3]
# print(ddd[["lat", "lon", "label", "tid", "time", "poi"]])
# print(val[["lat", "lon", "label", "tid", "time", "poi"]])
# print(ddd.shape)
# ddd["coord"] = ddd.apply(lambda x: (x["lat"],x["lon"]), axis=1)
# print(ddd["coord"].unique().shape)

print("Done!")