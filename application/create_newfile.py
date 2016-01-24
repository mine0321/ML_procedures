
# # -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class Create(object):
    def __init__(self, folder_name, old_filename):
        self.folder_name = folder_name
        self.old_filename = old_filename

    def create_newfile(self):
        folder_name = self.folder_name
        old_filename = self.old_filename
        if 'rri' in old_filename:
            sensor = 'rri'
            data = self.load_data(sensor, folder_name, old_filename)
            self.create_rri(folder_name, data)
        elif 'acce' in old_filename:
            sensor = 'acce'
            data = self.load_data(sensor, folder_name, old_filename)
            self.create_acce(folder_name, data)

    def create_acce(self, folder_name, lines):
        route = '%s/data/acce/%s_acce_undersheet.txt' % (
            folder_name, folder_name)
        data = []
        for line in lines:
            line = [datum.replace('$ACC ', '').replace(
                ' ', ',') for datum in line.split('\n') if '$ACC' in datum]
            # リストの中の改行を区切りとして、$ACCを''に、' 'を','に置き換える
            line.append('')  # ''を追加
            if not line == ['']:
                # ,を区切りとしたlineリストの最初の値に浮動小数点表示にする
                line = map(float, line[0].split(','))
                line.insert(1, np.nan)
                line.insert(1, np.nan)
                line.insert(1, np.nan)
                line.extend([np.nan, np.nan])
                data.append(line)  # data[]にlineをいれて数値データにする
        data = pd.DataFrame(data)
        data.to_csv(route, header=None, index=None)  # csvファイルで書き出し

    def load_data(self, sensor, folder_name, old_filename):
        route = '%s/data/%s/%s' % (folder_name, sensor, old_filename)
        f = open(route, 'r')
        lines = f.readlines()  # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
        f.close()
        return lines

    def create_rri(self, folder_name, data):
        route = '%s/data/rri/%s_rri.txt' % (folder_name, folder_name)
        f = open(route, 'w')
        for i, line in enumerate(data):
            line = [datum.replace('\r', '') for datum in line.split('\n')]
            line = line[0].split(' ')
            time = line[0].split(':')
            line[0] = float(time[0]) * 60 * 60 + float(time[1]) * 60 + float(time[2])
            line[1] = float(float(line[1]) / 1000)
            f.writelines(str(line[0]) + ' ' + str(line[1]) + '\n')
        f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process task information.')
    parser.add_argument(
        'folder_name', type=str, help='folder name of the data (yymmdd_name)')
    parser.add_argument(
        'old_filename', type=str, help='folder name of the old data')

    args = parser.parse_args()

    document = Create(args.folder_name, args.old_filename)
    document.create_newfile()
