import unittest
import pandas as pd
from utils import *
from data import *


class TestCorpus(unittest.TestCase):
    def test_read_corpus_file(self):
        path = "/home/masoumeh/Desktop/Thesis/BodyGesturePatternDetection/test/files/fullVideosClean.csv"
        output = read_csv_file(path)
        self.assertTrue(isinstance(output, pd.DataFrame))

    # def test_sort_corpus_time_file(self):
    #     path_raw = "/home/masoumeh/Desktop/MasterThesis/Data/fullVideosClean_with_SemanticType.csv"
    #     path_clean = "/home/masoumeh/Desktop/MasterThesis/Data/dataset_3.csv"
    #     df = read_csv_file(path_raw)
    #     dataset = read_csv_file(path_clean)
    #
    #     dff = pre_process(df, ["Unnamed: 0", "typePoint", "people"])
    #
    #     dff_ = dff[(dff.tid == 0) & (dff.time >= 0.95)]
    #     original = dff_[["point", "time", "words"]]
    #     print(original)
    #     expected = dff_[["point", "time", "words"]].sort_values(by="")
    #     print(expected)
    #
    #     dataset_ = dataset[(dataset.tid == 0) & (dataset.time >= 0.95)]
    #     output = dataset_[["point", "time", "words"]]
    #     print(output)
    #
    #     self.assertTrue(pd.testing.assert_frame_equal(expected, output))