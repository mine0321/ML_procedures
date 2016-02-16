
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import math

class Regression(object):
    def __init__(self, test_folder):
        self.test_folder = test_folder
        self.sensor_name = "acce_undersheet"
        self.gps_sensor = "acce_coat"
        self.data_col = 'acce_y'
        self.degree = 40
        self.max_dims = 100

    def regression(self):
        push_df = self.load_push(self.test_folder)
        error_df = self.load_timesync(self.test_folder)
        origin_target, origin_time = self.create_target(
            self.test_folder, push_df, error_df)
        origin_data, origin_gps = self.create_data(
            self.test_folder, push_df, error_df)
        scores = []
        for dim in xrange(2, self.max_dims + 1):
            print("-----  dimension is %d  ----" % dim)
            target, data, time, lat, lng = self.fit_dim(
                dim, origin_target, origin_data, origin_time,
                origin_gps.T[0], origin_gps.T[1])
            prd_target = self.cross_val(target, data)
            scores.append([dim, self.print_score(target, prd_target)])
        scores = np.array(scores).T
        return scores

    def training(self, target, data):
        clf = linear_model.LinearRegression()
        clf.fit(data, target)
        return clf

    def test(self, clf, data):
        return clf.predict(data)

    def cross_val(self, target, data, n_folds=10):
        rs = KFold(len(target), n_folds=n_folds, shuffle=True, random_state=0)
        prd = np.zeros(len(target))
        for train_ind, test_ind in rs:
            clf = linear_model.LinearRegression()
            clf.fit(data[train_ind], target[train_ind])
            prd[test_ind] = clf.predict(data[test_ind])
        return prd

    def print_score(self, true, predict):
        mae = mean_absolute_error(true, predict)
        print "mae : %f" % mae
        return mae

    def fit_dim(self, dim, target, data, time, lat, lng):
        data = np.array([data[ind:ind + dim] for ind in xrange(
            len(data[:-dim + 1]))])
        target = target[dim - 1:]
        time = time[dim - 1:]
        lat = lat[dim - 1:]
        lng = lng[dim - 1:]
        return target, data, time, lat, lng

    def create_target(self, folder_name, push_df, error_df):
        bpm_df = self.load_rri(folder_name, error_df)
        samples = self.sampling_labeled_data(bpm_df, push_df)
        sample_df = self.average_bpm(samples)
        sample_df = self.calculate_hrr(sample_df, bpm_df, folder_name)
        return np.array(sample_df.HRR), np.array(sample_df.time)

    def create_data(self, folder_name, push_df, error_df):
        base_df = self.load_basetime(folder_name)
        acce_df = self.load_acce(
            folder_name, self.sensor_name, error_df, base_df)
        gps_df = self.load_acce(
            folder_name, self.gps_sensor, error_df, base_df)
        acce_samples = self.sampling_labeled_data(acce_df, push_df)
        gps_samples = self.sampling_labeled_data(gps_df, push_df)
        sample_df = self.calc_p_p(acce_samples, gps_samples)
        return np.array(sample_df.p_p), np.array(sample_df[['lat', 'lng']])

    def norm_data(self, data):
        data = stats.zscore(np.array(data), axis=0)
        return data

    def calc_p_p(self, acce_samples, gps_samples):
        data = [[
            sample.time.mean(),
            max(sample[self.data_col] - min(sample[self.data_col])),
            gps_samples[ind].lat.mean(),
            gps_samples[ind].lng.mean()
        ] for ind, sample in enumerate(acce_samples)]
        df = pd.DataFrame(data, columns=['time', 'p_p', 'lat', 'lng'])
        return df

    def average_bpm(self, samples):
        data = [
            [sample.time.mean(), sample.bpm.mean()] for sample in samples]
        df = pd.DataFrame(data, columns=['time', 'bpm'])
        return df

    def sampling_labeled_data(self, data, label_df):
        """
    計測データからラベル時間毎に切り取ったデータを作成

    data : 計測データ
    label_df : ラベルデータ(pandas.DataFrame)
        """
        samples = []
        for ind, label in label_df.iterrows():
            samples.append(self.data_between(
                data, float(label.starttime), float(label.finishtime)))
        return samples

    def data_between(self, data, start, stop, column='time'):
        """
    startとstopの間のデータだけスライスする

    data : スライスしたいデータ(pandas.DataFrame)
    start : スライスの始点
    stop : スライスの終点
    column : dataのスライスするカラム
        """
        return data[(start <= data[column]) & (data[column] < stop)]

    def load_push(self, folder_name):
        return pd.read_csv(file('%s/data/%s_push.csv' % (
            folder_name, folder_name)), header=0)

    def estimate_mhr(self, age, sex):
        if sex == 'Male':
            return 220 - age
        elif sex == 'Female':
            return 206 - (0.88 * age)

    def find_minbpm(self, df):
        return df.bpm.min()

    def calculate_hrr(self, sample_df, bpm_df, folder_name):
        age, sex = self.load_participant(folder_name)
        maxbpm = self.estimate_mhr(age, sex)
        minbpm = self.find_minbpm(bpm_df)
        sample_df['HRR'] = (sample_df.bpm - minbpm) / (maxbpm - minbpm) * 100
        return sample_df

    def load_participant(self, folder_name):
        df = pd.read_csv(file('subjects_data.csv'))
        df = df.fillna(
            {'foldername': df.date.astype(np.string_) + "_" + df.name})
        age, sex = df[
            df.foldername == folder_name].ix[:, ('age', 'sex')].values[0]
        return age, sex

    def load_rri(self, folder_name, error_df, sumpling_time=0.01):
        f = open('%s/data/rri/%s_rri.txt' % (folder_name, folder_name))
        lines = f.readlines()  # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
        f.close()

        # 呼吸等のアーティファクタやエラーを取り除くために変動の範囲を設定
        min_rate = 1.1  # RRI(x)に対してRRI(x-1),RRI(x-2),RRI(x-3)からの低下率の許容範囲
        max_rate = 1.2  # RRI(x)に対してRRI(x-1),RRI(x-2),RRI(x-3)からの上昇率の許容範囲

        data = []
        rris = np.zeros(3)

        for i, line in enumerate(lines):
            line = [datum.replace('\r', '') for datum in line.split('\n')]
            line = line[0].split(' ')

            line[1] = float(line[1]) * 1000
            line[0] = float(line[0]) - float(error_df.heartrate)

            for j, pre_rri in enumerate(rris):
                if (pre_rri / min_rate > line[1]) or (
                        line[1] > pre_rri * max_rate):
                    break
                elif j == 2:
                    data.append(line)
            rris = np.delete(rris, 0)
            rris = np.append(rris, line[1])

        columns = ['time', 'RRI']
        data = pd.DataFrame(data, columns=columns)

        data = self.rolling_average(data, degree=self.degree).dropna()

        # 以下で取り除いたデータをスプライン補間によって補う．
        # リサンプリングタイムはsumpling_time秒

        ius = InterpolatedUnivariateSpline(data.time, data.RRI)
        xn = np.arange(
            int(data.head(1).time), data.tail(1).time, sumpling_time)
        yn = 60000 / ius(xn)
        data = pd.DataFrame({'time': pd.Series(xn), 'bpm': pd.Series(yn)})

        return data

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
            'acce_x', 'acce_y', 'acce_z', 'lat', 'lng'
        ]
        sub_cols = ['acce_x', 'acce_y', 'acce_z']

        df = pd.read_csv(file('%s/data/acce/%s_%s.csv' % (
            folder_name, folder_name, sensor_name)), names=columns)
        df.time = df.time - df.time[0] - error_df[sensor_name].values
        df = self.sub_data(df, sensor_name, sub_cols, base_df)
        df = self.rolling_average(df, degree=20)
        return df

    def load_timesync(self, folder_name):
        return pd.read_csv(file('%s/data/%s_timesync.csv' % (
            folder_name, folder_name)), header=0)

    def load_basetime(self, folder_name):
        return pd.read_csv(file('%s/data/%s_basetime.csv' % (
            folder_name, folder_name)), header=0, index_col=0)

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

    folder_list = ['20160116_hoshino', '20160112_araya']

    fig = plt.figure(figsize=[20, 5])
    ax = fig.add_subplot(111)

    for folder in folder_list:
        document = Regression(folder)
        scores = document.regression()
        ax.plot(scores[0], scores[1])
    fig.show()
