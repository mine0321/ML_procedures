# coding:utf-8

import pylab as plt
import pandas as pd
import numpy as np
import scipy.fftpack


def surround_by_box(x_start, x_stop, y_start, y_stop, ax, color = 'red'):
    ""
    "漕ぎ時間を四角形で囲う"
    ""
    interval1 = (x_start, x_stop)  # TODO: Chenge it later.意味ないので本実行の時に置換してまとめる
    ax.plot([interval1[0], interval1[0]],[y_start, y_stop], color=color)  # Left
    ax.plot([interval1[1], interval1[1]],[y_start, y_stop], color=color)  # Right
    ax.plot([interval1[0], interval1[1]],[y_start, y_start], color=color)  # Bottom
    ax.plot([interval1[0], interval1[1]],[y_stop, y_stop], color=color)  # Top
