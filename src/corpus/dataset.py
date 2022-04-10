from corpus.data import read_csv_file, create_train_test


# ----------------------------------------------Loading corpus-----------------------------------------------
df_clean = read_csv_file("/mnt/shared/people/masoumeh/MA/data/df_clean_data.csv")

root = "/mnt/shared/people/masoumeh/MA/data/datasets/"

set_names = ['head', 'hands', 'head_hands', 'all']
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
tasks = ['multi', 'bi_dem', 'bi_seq', 'bi_dei']
features = [["nx", "ny", "poi", "time"], ["x", "y", "poi", "time"]]
model_names = ['base', 'nn']
head_points = [0, 15, 16, 17, 18]
hands_points = [2, 3, 4, 5, 6, 7]
head_hands_points = [2, 3, 4, 5, 6, 7, 0, 15, 16, 17, 18]
all_points = None

keep_points = []
for set_name in set_names:
    for t_size in test_sizes:
        for task in tasks:
            for model_name in model_names:
                for feature in features:
                    if feature[0] == 'nx':
                        f_type = 'norm'
                    else:
                        f_type = 'unnorm'
                    if set_name == 'head':
                        print('set name is : ', set_name)
                        keep_points = head_points
                    elif set_name == 'hands':
                        print('set name is : ', set_name)
                        keep_points = hands_points
                    elif set_name == 'head_hands':
                        print('set name is : ', set_name)
                        keep_points = head_hands_points
                    else:
                        print('set name is ', str(set_name))
                        keep_points = None
                    create_train_test(df_clean,
                                      setname=set_name,
                                      test_size=t_size,
                                      with_pad=True,
                                      pad_value=99999,
                                      feature_cols=feature,
                                      features_type=f_type,
                                      class_col=task,
                                      keep_points=keep_points,
                                      model=model_name,
                                      root=root)





#
# # ----------------------------------------------Base Lines  -----------------------------------------------
# # data set 1
# feature_cols = ["x", "y", "poi", "time"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "base"
# fname_prefix = "set_1"
# class_col = "classes"
# # class_col = "biClasses"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"base/")
#
#
# # data set 2: using nx, ny
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "base"
# fname_prefix = "set_2"
# class_col = "classes"
# # class_col = "biClasses"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"base/")
#
#
# # data set 3: adding more body points
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "base"
# fname_prefix = "set_3"
# class_col = "classes"
# # class_col = "biClasses"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"base/")
#
#
# # data set 4: keep all points
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = None  # None means we do not exclude any point and keep all the points.
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "base"
# fname_prefix = "set_4"
# class_col = "classes"
# # class_col = "biClasses"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"base/")
#
#
# # data set 5: making the test size more
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.4
# with_pad = True
# pad_value = -98745
# model_name = "base"
# fname_prefix = "set_5"
# class_col = "classes"
# # class_col = "biClasses"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"base/")
#
#
# # data set 6: features without time
# feature_cols = ["nx", "ny", "poi"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.4
# with_pad = True
# pad_value = -98745
# model_name = "base"
# fname_prefix = "set_6"
# class_col = "classes"
# # class_col = "biClasses"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"base/")
#
#
# # ----------------------------------------------Neural Networks-----------------------------------------------
#
# # data set 1
# feature_cols = ["x", "y", "poi", "time"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.3
# with_pad = True
# pad_value = -10
# model_name = "nn"
# fname_prefix = "set_1"
# # class_col = "classes"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 2: using nx, ny
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_2"
# # class_col = "classes"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 3: adding more body points
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_3"
# # class_col = "classes"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 4: keep all points
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = None  # None means we do not exclude any point and keep all the points.
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_4"
# # class_col = "classes"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 5: making the test size more
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.4
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_5"
# # class_col = "classes"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
#
# # data set 6: features without time
# feature_cols = ["nx", "ny", "poi"]
# keep_points = [2, 3, 4, 5, 6, 7]
# test_size = 0.4
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_6"
# # class_col = "classes"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # Special only for bi_dem
# # data set 7: features
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
# test_size = 0.4
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_7"
# # class_col = "classes"
# class_col = "biClasses_dem"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
#
# # ----------------------------Only heads ------------------------------------------
#
# # data set 8: normalized features + 0.3 test size + multi
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_8"
# class_col = "classes"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/")
#
# # data set 9: unnormalized features + 0.3 test size + multi
# feature_cols = ["x", "y", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_9"
# class_col = "classes"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/")
#
#
# # data set 10: unnormalized features + 0.3 test size + binary(demarcative vs others)
# feature_cols = ["x", "y", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_10"
# class_col = "biClasses_dem"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 11: unnormalized features + 0.3 test size + binary(sequential vs others)
# feature_cols = ["x", "y", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_11"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 12: unnormalized features + 0.3 test size + binary(deictic vs others)
# feature_cols = ["x", "y", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_12"
# class_col = "biClasses_dei"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 13: normalized features + 0.3 test size + binary(demarcative vs others)
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_13"
# class_col = "biClasses_dem"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 14: normalized features + 0.3 test size + binary(sequential vs others)
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_14"
# class_col = "biClasses_seq"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
#
# # data set 15: normalized features + 0.3 test size + binary(deictic vs others)
# feature_cols = ["nx", "ny", "poi", "time"]
# keep_points = [0, 15, 16, 17, 18]
# test_size = 0.3
# with_pad = True
# pad_value = -98745
# model_name = "nn"
# fname_prefix = "set_15"
# class_col = "biClasses_dei"
# create_train_test(df_clean,
#                       test_size=test_size,
#                       with_pad=with_pad,
#                       pad_value=pad_value,
#                       feature_cols=feature_cols,
#                       class_col=class_col,
#                       keep_points=keep_points,
#                       model=model_name,
#                       fname_prefix=fname_prefix,
#                       path_to_save=path_to_save+"nn/binary/")
#
