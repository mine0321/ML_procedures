# coding:utf-8

import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def acceTocsv(filename, base_time, sync_error, experiment_time):

    """ 
    加速度txtデータをDataFrameに変換し、csvとして出力

    filename : 変換したいtxtデータのファイル名
    base_time : 時刻合わせに利用したセンサのbasetime
    sync_error : 実験開始時刻を0とするための時刻合わせ
    experiment_time : 実験時間を記述
    """

    f = open('%s.txt'%(filename)) # %Sというテキストファイルを開ける
    lines = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()
    data = pd.DataFrame({'time':[],'x':[],'y':[],'z':[]})
    # lines: リスト。要素は1行の文字列データ

    data = []

    for line in lines:
        line = [datum.replace('$ACC ', '').replace(' ', ',') for datum in line.split('\n') if '$ACC' in datum] 
        # リストの中の改行を区切りとして、$ACCを''に、' 'を','に置き換える
        line.append('')# ''を追加
        if not line == ['']:
            line = map(float, line[0].split(','))# ,を区切りとしたlineリストの最初の値に浮動小数点表示にする
            line[0] = line[0] - base_time - sync_error# timeの計算
            if 0 < line[0] < experiment_time:
                data.append(line)# data[]にlineをいれて数値データにする
    columns=['time', 'x', 'y', 'z']# columnsで表にする
    data = pd.DataFrame(data, columns=columns)
    columns.remove('time')

    for col in columns:
        data[col] = data[col] - data[col].mean()# センサの傾きの調整

    data.to_csv('%s.csv'%(filename))# csvファイルで書き出し
    return data

def rriTocsv(filename, sync_error, experiment_time, bpm=False,
    sumpling_time=1, ver_old=False):

    """
    心拍txtデータをDataFrameに変換し、csvとして出力

    filename : 変換したいtxtデータのファイル名
    sync_error : 実験開始時刻を0とするための時刻合わせ
    experiment_time : 実験時間を記述
    bpm : if True, bpm. if True, RRI
    sumpling_time : 補間時のresampling timeの指定
    ver_old : 2014年購入の心拍系の場合Trueにする

    """

    f = open('%s.txt'%(filename))
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

        if ver_old:
            time = line[0].split(':')
            line[0] = float(time[0])*60*60 + float(time[1])*60 + float(time[2]) - sync_error
            line[1] = float(line[1])
        else:
            line[1] = float(line[1]) * 1000
            line[0] = float(line[0]) - sync_error

        if 0 <= line[0] <= experiment_time:
            for j,pre_rri in enumerate(rris):
                if pre_rri/min_rate > line[1] or line[1] > pre_rri*max_rate:
                    break
                elif j == 2:
                    data.append(line)
        rris = np.delete(rris, 0)
        rris = np.append(rris, line[1])

    columns = ['time', 'RRI']
    data = pd.DataFrame(data, columns=columns)

    # 以下で取り除いたデータをスプライン補間によって補う．
    # リサンプリングタイムはsumpling_time秒

    ius = InterpolatedUnivariateSpline(data.time, data.RRI)
    xn  = np.arange(int(data.head(1).time), data.tail(1).time, sumpling_time)
    yn  = ius(xn)

    if bpm:
        data = pd.DataFrame({'time':pd.Series(xn),'RRI':pd.Series(yn)})
    else:
        yn = 60000 / yn
        data = pd.DataFrame({'time':pd.Series(xn),'bpm':pd.Series(yn)})

    data.to_csv('%s.csv'%(filename))

    return data

def emgTocsv(filename, sync_error):
    """
    筋電csvデータをDataFrameに変換し、csvとして出力

    # filename : 変換したいcsvデータのファイル名
    # sync_error : 実験開始時刻を0とするための時刻合わせ
    """
    columns = ['time', 'x', 'y', 'z', 'EMG1', 'EMG2', 'IEMG1', 'IEMG2']
    df = pd.read_csv(file('%s.csv'%(filename)), skiprows=10,
        names=columns)
    df['time'] = df['time']/1000 - sync_error

    columns.remove('time')

    for col in columns:
        df[col] = df[col] - df[col].mean()
    df.to_csv('%s_A.csv'%(filename))
    return df

def get_label_df(push_label,sync_error=0,min_time=0.5,max_time=1.5):
    """
    漕ぎラベルcsvをDataFrameへ変換

    push_label : csvデータのファイル名(漕ぎ開始を'StartTime'，漕ぎ終了を'FinishTime')
    sync_error : 実験開始時刻を0とするための時刻合わせ
    min_time : 短すぎる漕ぎをフィルタリング
    max_time : 長すぎる漕ぎをフィルタリング
    """
    df = pd.read_csv(file('%s.csv'%(push_label)), header=0)
    df = df[(min_time <= df['FinishTime'] - df['StartTime']) & (df['FinishTime'] - df['StartTime'] <= max_time)]
    df.StartTime = df.StartTime - sync_error
    df.FinishTime = df.FinishTime - sync_error
    return df

def sampling_labeled_data(data, label_df):
    """
    計測データからラベル時間毎に切り取ったデータを作成

    data : 計測データ
    label_df : ラベルデータ(pandas.DataFrame)
    """
    samples = []
    for ind, label in label_df.iterrows():
        samples.append(data_between(data, label.StartTime, label.FinishTime))
    return samples

def data_between(data, start, stop):
    return data[(start <= data.time) & (data.time < stop)]

def select_file(filename, sync_error, experiment_time,
    base_time=0, bpm=0, sumpling_time=1, ver_old=0):
    if 'acce' in filename:
        df = acceTocsv(filename, base_time, sync_error, experiment_time)
    elif 'RRI' in filename:
        df = rriTocsv(filename, sync_error, experiment_time, bpm, sumpling_time, ver_old)
    elif 'muscle' in filename:
        df = emgTocsv(filename, sync_error)
    return df
