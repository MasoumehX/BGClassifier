from data import create_data_train_test
from utils import write_csv_file

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
dataset, train, test = create_data_train_test(path_csv=csv_file_path, path_txt=txt_file_path, drop_cols=drop_cols,
                                        keep_point=keep_points, keep_typePoint=keep_typepoint,
                                        keep_people=keep_people)



# ---------------------------------------------------------------------------------------------#

# Writing the results in csv files
where_to_write = "/home/masoumeh/Desktop/MasterThesis/Data/"

# server
# where_to_write = "/data/home/masoumeh/Data/"
write_csv_file(dataset, path=where_to_write+"dataset_big_clean.csv")
write_csv_file(train, path=where_to_write+"train_big_clean.csv")
write_csv_file(test, path=where_to_write+"test_big_clean.csv")

print("Done!")
