
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Sampling(object):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.starttime = 5
        self.finish = 5
        self.num = 0.2
        self.border = 0.03
        self.min_sample_time = 0.5
        self.max_sample_time = 1.5
        self.axis = 'acce_y'
        self.sensor_name = "acce_undersheet"

    def run_sampling(self):
        acce_df, error_df = self.load_data(
            self.folder_name, self.sensor_name)
        self.finishtime = float(error_df.finish) - self.finish
        push_df = self.sampling(acce_df)
        self.plot_push(push_df, acce_df)
        self.output_push(push_df)

    def sampling(self, df):
        pushs = self.find_pushs(df)
        return self.convert_df(pushs)

    def find_pushs(self, df):
        ind = df.time.abs().idxmin()
        pushs = []
        while df.time[ind] < self.finishtime:
            maxvalue = df[self.axis][ind]
            ind = self.find_min(df, ind)
            minvalue = df[self.axis][ind]
            ind = self.find_max(df, ind)
            if maxvalue < df[self.axis][ind]:
                maxvalue = df[self.axis][ind]
            p_to_p = maxvalue - minvalue
            if p_to_p > self.border:
                pushs.append(df.time[ind])
        return pushs

    def find_min(self, df, ind):
        while True:
            pre_ind = ind
            search_range = [
                float(df.time[ind]), float(df.time[ind]) + self.num]
            ind = df[self.axis][(search_range[0] <= df.time) & (
                df.time <= search_range[1])].idxmin()
            if pre_ind == ind:
                return ind

    def find_max(self, df, ind):
        while True:
            pre_ind = ind
            search_range = [
                float(df.time[ind]), float(df.time[ind]) + self.num]
            ind = df[self.axis][(search_range[0] <= df.time) & (
                df.time <= search_range[1])].idxmax()
            if pre_ind == ind:
                return ind

    def plot_push(self, push_df, acce_df):
        fig = plt.figure(figsize=[13, 7])
        ax = fig.add_subplot(111)

        x_axis = 'time'
        value = 0.05
        ylim = [-0.2, 0.2]
        xlim = [self.starttime, self.finishtime]
        interval = np.array(ylim) + np.array([value, -value])
        acce_df.plot(x=x_axis, y=self.axis, ylim=ylim, xlim=xlim, ax=ax)
        self.plot_label(push_df, interval, ax=ax)
        fig.show()

    def output_push(self, df):
        df.to_csv('%s/data/%s_push.csv' % (
            self.folder_name, self.folder_name), index=None)

    def convert_df(self, pushs):
        push_df = pd.DataFrame({
            "starttime": pd.Series(pushs[:-1]),
            "finishtime": pd.Series(pushs[1:])})
        push_df = self.fix_df(push_df)
        return push_df

    def fix_df(self, df):
        min_time = self.min_sample_time
        max_time = self.max_sample_time
        df = df[(min_time <= df.finishtime - df.starttime) & (
            df.finishtime - df.starttime <= max_time)]
        return df

    def load_data(self, folder_name, sensor_name):
        error_df = self.load_timesync(folder_name)
        base_df = self.load_basetime(folder_name)
        acce_df = self.load_acce(folder_name, sensor_name, error_df, base_df)
        return acce_df, error_df

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

    def sub_data(self, df, sensor_name, cols, base_df):
        for col in cols:
            initial = float(base_df[sensor_name].starttime)
            last = float(base_df[sensor_name].finishtime)
            base = df[col][(initial < df.time) & (df.time < last)].mean()
            df[col] = df[col] - base
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
