
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


class AllPlotters(object):
    def __init__(self, folder_name, y_axis, xlim, ylim, emg, check):
        self.folder_name = folder_name
        self.y_axis = y_axis
        self.xlim = xlim
        self.ylim = ylim
        self.emg = emg
        self.check = check

    def acce_plot(self):
        if self.check:
            error_df = self.load_timesync(self.folder_name)
        else:
            error_df = None
        dfs = self.load_acces(self.folder_name, error_df)
        self.plot_acce_dfs(dfs, self.y_axis, self.xlim, self.ylim)

    def sync_of_emg_plot(self):
        if self.check:
            error_df = self.load_timesync(self.folder_name)
        else:
            error_df = None
        dfs = self.load_one_acce(self.folder_name, error_df)
        dfs.append(self.load_emg(self.folder_name, error_df))
        self.plot_acce_for_sync_emg(dfs, self.y_axis, self.xlim, self.ylim)

    def load_one_acce(self, folder_name, error_df):
        columns = [
            'time', 'gyro_x', 'gyro_y', 'gyro_z',
            'acce_x', 'acce_y', 'acce_z', 'lat', 'long'
        ]
        place = "backpack"

        df = pd.read_csv(file('%s/data/acce/%s_acce_%s.csv' % (
            folder_name, folder_name, place)), names=columns)
        if error_df is None:
            df.time = df.time - df.time[0]
        else:
            df.time = df.time - df.time[0] - error_df["acce_%s" % place].values
        df.acce_x = df.acce_x * 1000
        df.acce_y = df.acce_y * 1000
        df.acce_z = df.acce_z * 1000
        return [df]

    def load_timesync(self, folder_name):
        return pd.read_csv(file('%s/data/%s_timesync.csv' % (
            folder_name, folder_name)), header=0)

    def load_acces(self, folder_name, error_df):
        columns = [
            'time', 'gyro_x', 'gyro_y', 'gyro_z',
            'acce_x', 'acce_y', 'acce_z', 'lat', 'long'
        ]
        places = ["undersheet", "backpack", "coat", "sheetpocket"]
        dfs = []

        for place in places:
            df = pd.read_csv(file('%s/data/acce/%s_acce_%s.csv' % (
                folder_name, folder_name, place)), names=columns)
            if error_df is None:
                df.time = df.time - df.time[0]
            else:
                df.time = df.time - df.time[0] - error_df[
                    "acce_%s" % place].values
            df.acce_x = df.acce_x * 1000
            df.acce_y = df.acce_y * 1000
            df.acce_z = df.acce_z * 1000
            dfs.append(df)
        return dfs

    def load_emg(self, folder_name, error_df):
        columns = [
            'time', 'acce_x', 'acce_y', 'acce_z',
            'EMG1', 'EMG2', 'IEMG1', 'IEMG2'
        ]
        emg_df = pd.read_csv(file('%s/data/emg/%s_emg.csv' % (
            folder_name, folder_name)), skiprows=10, names=columns)
        if error_df is None:
            emg_df['time'] = emg_df['time'] / 1000
        else:
            emg_df['time'] = emg_df['time'] / 1000 - error_df.emg.values
        return emg_df

    def plot_acce_dfs(self, dfs, y_axis, xlim, ylim):
        fig = plt.figure(figsize=[15, 7])
        ax = fig.add_subplot(111)
        x_axis = 'time'

        for df in dfs:
            df.plot(
                x=x_axis, y=y_axis, ax=ax, xlim=xlim, ylim=ylim, legend=False)

        legends = ["undersheet", "backpack", "coat", "sheetpocket"]
        plt.legend(legends)
        plt.show()

    def plot_acce_for_sync_emg(self, dfs, y_axis, xlim, ylim):
        x_axis = 'time'
        legends = ["backpack", "emg"]

        if self.check:
            fig = plt.figure(figsize=[15, 3])
            ax = fig.add_subplot(111)
            for df in dfs:
                df.plot(
                    x=x_axis, y=y_axis, ax=ax,
                    xlim=xlim, ylim=ylim, legend=False)
            plt.legend(legends)
        else:
            for df, legend in zip(dfs, legends):
                fig = plt.figure(figsize=[15, 3])
                ax = fig.add_subplot(111)
                df.plot(
                    x=x_axis, y=y_axis, ax=ax,
                    xlim=xlim, ylim=ylim, legend=False)
                plt.legend([legend])

        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process task information.')
    parser.add_argument(
        'folder_name', type=str, help='folder name of the data (yymmdd_name)')
    parser.add_argument(
        '--y_axis', default='acce_y', type=str, help='for plot')
    parser.add_argument(
        '--xlim', default=None, help='for plot')
    parser.add_argument(
        '--ylim', default=None, help='for plot')
    parser.add_argument(
        '--emg', action="store_true", help='plot for sync of emg')
    parser.add_argument(
        '--check', action="store_true", help='use tymesync.csv')

    args = parser.parse_args()

    document = AllPlotters(
        args.folder_name, args.y_axis, args.xlim,
        args.ylim, args.emg, args.check)

    if args.emg:
        document.sync_of_emg_plot()
    else:
        document.acce_plot()
