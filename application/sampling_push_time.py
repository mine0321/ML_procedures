
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Sampling(object):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.min_search_time = 0.5
        self.max_search_time = 1
        self.max_first_push_time = 5
        self.min_last_push_time = 10
        self.min_sample_time = 0.5
        self.max_sample_time = 1.5
        self.timing = 0.1
        self.sensor_name = "acce_undersheet"
        self.emg_col = 'sum_EMGs'
        self.acce_col = 'acce_y'

    def run_sampling(self):
        acce_df, emg_df, error_df = self.load_data(
            self.folder_name, self.sensor_name)
        finish = float(error_df.finish)
        push_df = self.sampling(
            acce_df, emg_df, self.min_search_time, self.max_search_time,
            self.max_first_push_time, self.min_last_push_time,
            self.min_sample_time, self.max_sample_time, finish, self.timing
        )
        self.output_push(push_df, self.folder_name)
        self.plot_push(
            push_df, acce_df, emg_df, self.acce_col, self.emg_col, finish)

    def output_push(self, push_df, folder_name):
        push_df.to_csv('%s/data/%s_push.csv' % (folder_name, folder_name))

    def plot_push(self, push_df, acce_df, emg_df, acce_col, emg_col, finish):
        fig = plt.figure(figsize=[13, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        x_axis = 'time'
        xlim = [0, finish]

        value = 0.05
        ylim = [-0.2, 0.2]
        interval = np.array(ylim) + np.array([value, -value])
        acce_df.plot(x=x_axis, y=acce_col, ylim=ylim, xlim=xlim, ax=ax1)
        self.plot_label(push_df, interval, ax=ax1)

        value = 50
        ylim = [-50, 250]
        emg_df.plot(x=x_axis, y=emg_col, ylim=ylim, xlim=xlim, ax=ax2)
        interval = np.array(ylim) + np.array([value, -value])
        self.plot_label(push_df, interval, ax=ax2)

        plt.show()

    def sampling(
        self, acce_df, emg_df, min_search_time, max_search_time,
        max_first_push_time, min_last_push_time, min_sample_time,
        max_sample_time, finish, timing
    ):

        push = self.find_initialpush(
            emg_df, self.emg_col, max_first_push_time, min_search_time)
        finish_time = finish - min_last_push_time

        pushs = self.find_pushs_emg(
            push, emg_df, self.emg_col, finish_time,
            min_search_time, max_search_time)
        pushs = self.find_pushs_acce(pushs, timing, acce_df, self.acce_col)

        push_df = self.convert_df(pushs)
        push_df = self.fix_df(push_df, min_sample_time, max_sample_time)
        return push_df

    def fix_df(self, df, min_time, max_time):
        df = df[(min_time <= df.finishtime - df.starttime) & (
            df.finishtime - df.starttime <= max_time)]
        return df

    def convert_df(self, pushs):
        push_df = pd.DataFrame({
            "starttime": pd.Series(pushs[:-1]),
            "finishtime": pd.Series(pushs[1:])})
        return push_df

    def find_pushs_acce(self, pushs, timing, df, col):
        for push_ind, push in enumerate(pushs):
            search_range = [float(push - timing), float(push + timing)]
            ind = df[col][
                (search_range[0] <= df.time) & (
                    df.time <= search_range[1])].idxmax()
            pushs[push_ind] = df.time[ind]
        return pushs

    def find_pushs_emg(self, push, df, col, finish, min_time, max_time):
        pushs = [push]
        while pushs[-1] < finish:
            search_range = [
                float(pushs[-1] + min_time), float(pushs[-1] + max_time)]
            ind = df[col][(search_range[0] <= df.time) & (
                df.time <= search_range[1])].idxmin()
            ind = self.find_smaller(df, ind, col, min_time)
            pushs.append(df.time[ind])
        return pushs

    def find_initialpush(self, df, col, max_time, min_time):
        ind = df[col][(0 <= df.time) & (df.time <= max_time)].idxmin()
        ind = self.find_smaller(df, ind, col, min_time)
        return df.ix[ind].time

    def find_smaller(self, df, ind, col, min_time):
        pre_ind = None
        while pre_ind != ind:
            pre_ind = ind
            search_range = [
                float(df.time[ind]), float(df.time[ind]) + min_time]
            ind = df[col][(search_range[0] <= df.time) & (
                df.time <= search_range[1])].idxmin()
        return ind

    def sum_emgs(self, emg_df):
        emg_df['abs_EMG1'] = emg_df.EMG1.abs()
        emg_df['abs_EMG2'] = emg_df.EMG2.abs()
        emg_df['sum_EMGs'] = emg_df['abs_EMG1'] + emg_df['abs_EMG2']
        return emg_df

    def load_data(self, folder_name, sensor_name):
        error_df = self.load_timesync(folder_name)
        base_df = self.load_basetime(folder_name)
        acce_df = self.load_acce(folder_name, sensor_name, error_df, base_df)
        emg_df = self.load_emg(folder_name, error_df, base_df)
        return acce_df, emg_df, error_df

    def sub_data(self, df, sensor_name, cols, base_df):
        for col in cols:
            initial = float(base_df[sensor_name].starttime)
            last = float(base_df[sensor_name].finishtime)
            base = df[col][(initial < df.time) & (df.time < last)].mean()
            df[col] = df[col] - base
        return df

    def load_acce(self, folder_name, sensor_name, error_df, base_df):
        columns = [
            'time', 'gyro_x', 'gyro_y', 'gyro_z',
            'acce_x', 'acce_y', 'acce_z', 'lat', 'long'
        ]
        sub_cols = ['acce_x', 'acce_y', 'acce_z']

        df = pd.read_csv(file('%s/data/acce/%s_%s.csv' % (
            folder_name, folder_name, sensor_name)), names=columns)
        df.time = df.time - df.time[0] - error_df[sensor_name].values
        df = self.sub_data(df, sensor_name, sub_cols, base_df)
        df = self.rolling_average(df, degree=20)
        return df

    def load_emg(self, folder_name, error_df, base_df):
        columns = [
            'time', 'acce_x', 'acce_y', 'acce_z',
            'EMG1', 'EMG2', 'IEMG1', 'IEMG2'
        ]
        sub_cols = ['EMG1', 'EMG2']

        df = pd.read_csv(file('%s/data/emg/%s_emg.csv' % (
            folder_name, folder_name)), skiprows=10, names=columns)
        df.time = df.time / 1000 - error_df.emg.values
        df = self.sub_data(df, 'emg', sub_cols, base_df)
        df = self.sum_emgs(df)
        df = self.rolling_average(df, degree=200)
        return df

    def load_timesync(self, folder_name):
        return pd.read_csv(file('%s/data/%s_timesync.csv' % (
            folder_name, folder_name)), header=0)

    def load_basetime(self, folder_name):
        return pd.read_csv(file('%s/data/%s_basetime.csv' % (
            folder_name, folder_name)), header=0, index_col=0)

    def surround_by_box(
            self, x_start, x_stop, y_start, y_stop, ax=plt, color='red'):
        """
    四角形で囲う

    x_start : 横軸始点
    x_stop : 横軸終点
    y_start : 縦軸始点
    y_stop : 縦軸終点
    ax : 軸の指定
    color : 色の指定
        """
        # TODO: Chenge it later.意味ないので本実行の時に置換してまとめる
        interval1 = (x_start, x_stop)
        # Left
        ax.plot([interval1[0], interval1[0]], [y_start, y_stop], color=color)
        # Right
        ax.plot([interval1[1], interval1[1]], [y_start, y_stop], color=color)
        # Bottom
        ax.plot([interval1[0], interval1[1]], [y_start, y_start], color=color)
        # Top
        ax.plot([interval1[0], interval1[1]], [y_stop, y_stop], color=color)

    def plot_label(self, label_df, y_interval, ax=plt, color='red'):

        """
    データ波形に漕ぎ時間を四角形囲った波形をプロット

    label_df : 漕ぎ時間をpandas.DataFrameで挿入culumns =['StartTime','FinishTime']
    y_interval : 縦軸の始点と終点を指定[始点，終点]
    ax : 軸の指定
        """
        [y_start, y_stop] = y_interval
        for ind, label in label_df.iterrows():
            self.surround_by_box(
                label.starttime, label.finishtime,
                y_start, y_stop, ax, color=color
            )

    def rolling_average(self, df, degree=10):
        """
    入力された加速度データの移動平均をとって出力
    移動平均の計算の際には前後5サンプルをデフォルトで利用する
    （degreeの値を変更することで変更可能）

    デフォルトを10サンプルにしているのは経験則
        """
        df = pd.rolling_mean(df, degree, center=True)
        return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process task information.')
    parser.add_argument(
        'folder_name', type=str, help='folder name of the data (yymmdd_name)')

    args = parser.parse_args()

    document = Sampling(args.folder_name)
    document.run_sampling()
