import os
import numpy as np
from data import *
from utils import *
from plot import *
from collections import Counter





if __name__ == '__main__':
    root = "/home/masoumeh/Desktop/MasterThesis/Data/"
    df_val = read_csv_file(path=root + "val.csv")
    print(df_val.tid.unique())
    ddd = df_val[df_val.tid == 18]
    ddd["coord"] = ddd.apply(lambda x: (x["lat"], x["lon"]), axis=1)
    # print(ddd[["coord", "label", "tid", "time", "poi"]])
    coords = [i for i, row in enumerate(ddd["coord"].values)]
    ddd["y"] = coords
    print(len(coords))
    plot_line(y=ddd["time"], x=ddd["y"], filename="coord", title="Coord X Y ", show=True, save=False)
    # path = "/home/masoumeh/Desktop/MasterThesis/Data/"
    # fullvideoclean = "fullVideosClean_with_SemanticType.csv"
    # createtime = "CREATIME.xlsx"
    # df = read_csv_file(os.path.join(path, fullvideoclean))
    # df = pre_process(df)
    # # print(df.columns)
    # #
    # # sample = df[(df.words == "from_start_to_finish") &
    # #             (df["name"] == "2009-08-12_1400_US_KNBC_Today_Show_1259-1264_ID703_from_start_to_finish_json")]
    # # print(sample[["words", "SemanticType", "x", "y", "point", "frame"]])
    #
    #
    # df_right = df[df.point.isin([2, 3, 4])]  # right hands
    #
    #
    # # # normalizing
    # def norm_time(x):
    #     return (x - min(x)) / (max(x) - min(x))
    #
    #
    # df_right["time"] = list(df_right.groupby("name").frame.apply(norm_time))
    # print(df_right[["words", "x", "y", "point", "frame", "time"]])

    # df_right = df_right.sort_values(by="frame")

    # groups = df_right.groupby("name")
    # for name, group in groups:
    #     print(group[["x", "y", "point", "frame", "time", "words"]])
        # group["time"] = norm_time(group)

    # df_right["time"] = df_right.groupby("name").apply(lambda x: norm_time(x))
    # print(df_right)
    # res = list()
    # nms = df_right.name.unique()
    # cnt = 0
    # for n in nms:
    #     tmp = df_right[df_right.name == n]
    #     tmp["time"] = norm_time(tmp.frame)
    #     print(tmp)
    #     cnt = cnt + 1
    #     res[[cnt]]=tmp
    #
    # res2 = do.call(rbind, res)

    # print(sample.point.unique())
    # sample = sample[sample.point.isin([6, 7])]
    # plot_line(sample.x, sample.y, "line", "plot lines", False, True)
    #
    # plot_all_points_for_words(df)





    # token_1 = "from_start_to_finish"
    # token_2 = "from_beginning_to_end"
    # sample = df[(df.words == token_2)]

    # form start to finish: remove points [8,9,12,10,13,11,24,23,22,21,14,19,20]
    # from beginning to end: remove points [8,9,12,10,13,11,24,23,22,21,14,19,20]

    # sample = sample[~sample.point.isin(non_related_points)]
    # sample["point"] = sample["point"].astype(str)
    # groups = sample.groupby(by="name")
    # for name, group in groups:
    #     print(name)
    #     plot_scatter(group, x="x", y="y", labels="point", filename="scatter_plot_from_beginning_to_end_"+name,
    #                  title="Scatter plot of points for word from beginning to end")




    # print(len(labels))
    # print(coords)
    # print(start_to_finish[["labels", "coords"]])
    # sample = start_to_finish[start_to_finish["name"] == "2009-08-12_1400_US_KNBC_Today_Show_1259-1264_ID703_from_start_to_finish_json"]
    # print(sample.point.unique())
    # sample_coords_x = sample["x"].values
    # sample_coords_y = sample["y"].values
    # sample_labels = sample.point.values
    #
    # print(len(sample_labels))
    #
    # sample["x"] = sample_coords_x
    # sample["y"] = sample_coords_y
    # sample["label"] = sample_labels
    #
    # new_cords = sample[sample.label.isin([4, 3, 6, 7])]
    # print(new_cords)
    # plot_scatter(new_cords["x"].values, new_cords["y"].values, new_cords["label"].values)
    # # plot_scatter(sample_coords_x, sample_coords_y, sample_labels)



    # print(sample[["words", "x", "y", "point", "frame"]])
    # print(sample.point.unique())
    # groups = sample.groupby(by="point")
    # for n, g in groups:
    #     print(n, g)
    # print(df.loc[0]["name"])
    # print(df.loc[0])
    # print(df[df["discarded_len"] == 1])

    # print("clean data shape: ", df_clean.shape)  # 361191, 15
    # print(df_clean.columns)str.
    # print(df_clean.words.unique())
    # print(df_clean.SemanticType.unique())   # 3 semantic types
    # print(df_clean.words.unique().shape)    # 24 type of expressions

    # deictic = df_clean[df_clean.SemanticType == "deictic"]

    # print(deictic.words.unique())
    # deictic_past = deictic[deictic.words.str.contains('past')]
    # all_past = df_clean[df_clean["words"].str.contains('past', na=False)]
    # print(deictic_past.words.unique())
    # print(all_past.words.unique())


    # print_data_stats(df)
    # expressions, counts = get_freq_power(df_clean, "words")
    # print("Range: ", min(counts), " ... ", max(counts))
    # plot_freq_power(keys, values, save=True, title="raw")
    # df_sample = df_clean[df_clean.words == "from_beginning_to_end"]
    # fname = df_sample["name"].iloc[0]
    # print(fname)
    # df_sample_from_beginning_to_end = df_sample[df_sample["name"] == fname]
    # print(df_sample_from_beginning_to_end)
    # print(df_sample_from_beginning_to_end.point)