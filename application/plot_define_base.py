
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


class AllPlotters(object):
    def __init__(self, folder_name, sensor_name, check, run):
        self.folder_name = folder_name
        self.sensor_name = sensor_name
        self.check = check
        self.run = run

    def select_data(self):
        error_df = self.load_timesync(self.folder_name)
        if self.check:
            xlim = self.load_basetime(self.folder_name, self.sensor_name)
        else:
            xlim = None

        if 'emg' == self.sensor_name:
            df = self.load_emg(self.folder_name, error_df)
            if self.run:
                columns = ['EMG1', 'EMG2']
                time_range = self.load_basetime(
                    self.folder_name, self.sensor_name)
                df = self.sub_data(df, columns, time_range)
            self.plot_emg(df, xlim)
        elif "acce" in self.sensor_name:
            df = self.load_acce(self.folder_name, self.sensor_name, error_df)
            if self.run:
                columns = ['acce_x', 'acce_y', 'acce_z']
                time_range = self.load_basetime(
                    self.folder_name, self.sensor_name)
                df = self.sub_data(df, columns, time_range)
            self.plot_acce(df, xlim)

    def sub_data(self, df, cols, time_range):
        for col in cols:
            base = df[col][(float(time_range[0]) < df.time) & (df.time < float(time_range[1]))].mean()
            df[col] = df[col] - base
        return df

    def load_acce(self, folder_name, sensor_name, error_df):
        columns = [
            'time', 'gyro_x', 'gyro_y', 'gyro_z',
            'acce_x', 'acce_y', 'acce_z', 'lat', 'long'
        ]

        df = pd.read_csv(file('%s/data/acce/%s_%s.csv' % (
            folder_name, folder_name, sensor_name)), names=columns)
        df.time = df.time - df.time[0] - error_df[sensor_name].values
        return df

    def load_emg(self, folder_name, error_df):
        columns = [
            'time', 'acce_x', 'acce_y', 'acce_z',
            'EMG1', 'EMG2', 'IEMG1', 'IEMG2'
        ]
        df = pd.read_csv(file('%s/data/emg/%s_emg.csv' % (
            folder_name, folder_name)), skiprows=10, names=columns)
        df.time = df.time / 1000 - error_df.emg.values
        return df

    def plot_acce(self, df, xlim=None):
        fig = plt.figure(figsize=[15, 7])
        ax = fig.add_subplot(111)
        x_axis = 'time'
        y_axis = ['acce_x', 'acce_y', 'acce_z']
        df.plot(x=x_axis, y=y_axis, xlim=xlim, ax=ax)
        plt.show()

    def plot_emg(self, df, xlim):
        fig = plt.figure(figsize=[15, 7])
        ax = fig.add_subplot(111)
        x_axis = 'time'
        y_axis = ['EMG1', 'EMG2']
        df.plot(x=x_axis, y=y_axis, xlim=xlim, ax=ax)
        plt.show()

    def load_timesync(self, folder_name):
        return pd.read_csv(file('%s/data/%s_timesync.csv' % (
            folder_name, folder_name)), header=0)

    def load_basetime(self, folder_name, sensor_name):
        df = pd.read_csv(file('%s/data/%s_basetime.csv' % (
            folder_name, folder_name)), header=0, index_col=0)
        return [df[sensor_name].starttime, df[sensor_name].finishtime]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process task information.')
    parser.add_argument(
        'folder_name', type=str, help='folder name of the data (yymmdd_name)')
    parser.add_argument(
        'sensor_name', type=str,
        help='sensor name of the data ("acce_??"" or "emg")')
    parser.add_argument(
        '--check', action="store_true", help='plot using xlim by basetime.csv')
    parser.add_argument(
        '--run', action="store_true",
        help='plot after subtracting values of basetime.csv')

    args = parser.parse_args()

    document = AllPlotters(
        args.folder_name, args.sensor_name, args.check, args.run)
    document.select_data()
