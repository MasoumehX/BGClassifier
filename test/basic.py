import unittest
import pandas as pd
from utils import read_csv_file


class TestCorpus(unittest.TestCase):
    def test_read_corpus_file(self):
        path = "/home/masoumeh/Desktop/Thesis/BodyGesturePatternDetection/test/files/fullVideosClean.csv"
        output = read_csv_file(path)
        self.assertTrue(isinstance(output, pd.DataFrame))
