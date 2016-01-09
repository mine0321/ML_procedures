# coding:utf-8

import pylab as plt



def surround_by_box(x_start, x_stop, y_start, y_stop, ax=plt, color='red'):
    """
    四角形で囲う

    x_start : 横軸始点
    x_stop : 横軸終点
    y_start : 縦軸始点
    y_stop : 縦軸終点
    ax : 軸の指定
    color : 色の指定
    """
    interval1 = (x_start, x_stop)  # TODO: Chenge it later.意味ないので本実行の時に置換してまとめる
    ax.plot([interval1[0], interval1[0]], [y_start, y_stop], color=color)  # Left
    ax.plot([interval1[1], interval1[1]], [y_start, y_stop], color=color)  # Right
    ax.plot([interval1[0], interval1[1]], [y_start, y_start], color=color)  # Bottom
    ax.plot([interval1[0], interval1[1]], [y_stop, y_stop], color=color)  # Top

def plot_label(label_df, y_interval, ax=plt):
    """
    データ波形に漕ぎ時間を四角形囲った波形をプロット

    label_df : 漕ぎ時間をpandas.DataFrameで挿入culumns =['StartTime','FinishTime']
    y_interval : 縦軸の始点と終点を指定[始点，終点]
    ax : 軸の指定
    """
    [y_start, y_stop] = y_interval
    for ind, label in label_df.iterrows():
        surround_by_box(label.starttime, label.finishtime, y_start, y_stop, ax)