# coding:utf-8

import pylab as plt
import iwavelets.pycwt as pycwt
import pandas as pd
import numpy as np
import scipy.fftpack


def rolling_average(acce_df, degree=7):
    """
    入力された加速度データの移動平均をとって出力
    移動平均の計算の際には前後5サンプルをデフォルトで利用する
    （degreeの値を変更することで変更可能）

    デフォルトを5サンプルにしているのは経験則
    """
    import pandas as pd
    acce_df = pd.rolling_mean(acce_df, degree)
    return acce_df


def surround_by_box(x_start, x_stop, y_start, y_stop, color = 'red'):
    ""
    "漕ぎ時間を四角形で囲う"
    ""
    interval1 = (x_start, x_stop)  # TODO: Chenge it later.意味ないので本実行の時に置換してまとめる
    plt.plot([interval1[0], interval1[0]],[y_start, y_stop], color=color)  # Left
    plt.plot([interval1[1], interval1[1]],[y_start, y_stop], color=color)  # Right
    plt.plot([interval1[0], interval1[1]],[y_start, y_start], color=color)  # Bottom
    plt.plot([interval1[0], interval1[1]],[y_stop, y_stop], color=color)  # Top
