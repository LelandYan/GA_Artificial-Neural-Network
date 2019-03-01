# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/3/1 9:13'

import pandas as pd

def load_data(CSV_FILE_PATH='csv_result-colonTumor.csv'):
    df = pd.read_csv(CSV_FILE_PATH)
    shapes = df.values.shape
    # the eigenvalue of file
    input_data = df.values[:, 1:shapes[1] - 1]
    # the result of file
    result = df.values[:, shapes[1] - 1:shapes[1]]
    return input_data,result