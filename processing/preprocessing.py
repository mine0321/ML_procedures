# coding:utf-8

import pandas as pd
import numpy as np

def rolling_average(acce_df, degree=7):
    """
    入力された加速度データの移動平均をとって出力
    移動平均の計算の際には前後5サンプルをデフォルトで利用する
    （degreeの値を変更することで変更可能）

    デフォルトを5サンプルにしているのは経験則
    """
    acce_df = pd.rolling_mean(acce_df, degree, center=True).dropna()
    return acce_df

def calc_IEMG(emg_data, label_df,degree=100):
    """
    筋電データの積分値を整流後に計算

    emg_data : pandas形式の筋電データ
    label_df : pandas形式のlabelデータ

    ToDo
    * 整流→平滑化→計算と変更したい
    * 評価値として平均振幅を使用したほうがいいかも
    """

    import processing.load_data as load

    emg_data_ave = rolling_average(emg_data.abs(),degree=degree)
    samples = load.sampling_labeled_data(emg_data_ave, label_df)
    IEMG_list = [[sample.time.mean(), sum(sample.EMG1),
                            sum(sample.EMG2)] for sample in samples]
    return pd.DataFrame(IEMG_list, columns=['time', 'IEMG1', 'IEMG2'])

def pairwise_dtw(samples, axis):
    """
    DTWによるサンプルのペアワイズ距離を取る
    """
    import mlpy
    from scipy.spatial.distance import squareform
    array1 = map(lambda data: data[axis], samples)
    d_array = []
    l_array = len(array1)
    for i in range(l_array-1):
        for j in range(i+1, l_array):
            d_array.append(mlpy.dtw_std(array1[i], array1[j], dist_only=True))
    X = squareform(d_array)
    return X