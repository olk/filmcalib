'''
                    Copyright Oliver Kowalke 2020.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import abc
import argparse
import json
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator
from pathlib import Path
from prettytable import PrettyTable
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest


def outliers_mask(x, y):
    dy = np.gradient(y, x)
    rgr = IsolationForest()
    #rgr = IsolationForest(contamination=float(.20), random_state=42)
    rgr.fit(x.reshape(-1, 1), dy.reshape(-1, 1))
    dy_dt = rgr.predict(x.reshape(-1, 1)).flatten()
    msk = np.array([])
    for y in np.unique(dy_dt):
        msk_ = dy_dt == y
        # choose the list with most elements
        if (np.count_nonzero(msk) < np.count_nonzero(msk_)):
            msk = msk_
    return msk


class Model(abc.ABC):
    _msk = np.array([])

    @abc.abstractmethod
    def _process(self):
        'process data'

    def __init__(self, data):
        self._devtime, self._gamma, _ , _ = zip(*data)
        self._devtime = np.array(self._devtime)
        self._gamma = np.array(self._gamma)
        self._msk = outliers_mask(self._gamma, self._devtime)
        self._process()

    @abc.abstractmethod
    def model_name(self):
        'model name'

    @property
    def devtime(self):
        return self._devtime

    @property
    def gamma(self):
        return self._gamma

    @property
    def Nplus2(self):
        return self._Nplus2

    @property
    def Nplus1(self):
        return self._Nplus1

    @property
    def N(self):
        return self._N

    @property
    def Nminus1(self):
        return self._Nminus1

    @property
    def Nminus2(self):
        return self._Nminus2

    @property
    def r_square(self):
        return self._r2

    @property
    def msk(self):
        return self._msk

    def predict(self, x):
        return self._model.predict(x)


'metric based on: Lambrecht/Woodhouse, Way Beyond Monochrome'
class TheilSenRegressionModel(Model):
    def __init__(self, data):
        super(TheilSenRegressionModel, self).__init__(data)

    def _process(self):
        self._model = TheilSenRegressor().fit(self.gamma[self.msk].reshape(-1, 1), self.devtime[self.msk].reshape(-1, 1))
        self._r2 = round(self._model.score(self.gamma[self.msk].reshape(-1, 1), self.devtime[self.msk].reshape(-1, 1)), 3)
        self._Nplus2 = self.predict(np.array(0.8).reshape(-1, 1))[0]
        self._Nplus1 = self.predict(np.array(0.67).reshape(-1, 1))[0]
        self._N = self.predict(np.array(0.57).reshape(-1, 1))[0]
        self._Nminus1 = self.predict(np.array(0.5).reshape(-1, 1))[0]
        self._Nminus2 = self.predict(np.array(0.44).reshape(-1, 1))[0]

    def model_name(self):
        return __class__.__name__;


class RANSACRegressionModel(Model):
    def __init__(self, data):
        super(RANSACRegressionModel, self).__init__(data)

    def _process(self):
        self._model = RANSACRegressor().fit(self.gamma[self.msk].reshape(-1, 1), self.devtime[self.msk].reshape(-1, 1))
        self._r2 = round(self._model.score(self.gamma[self.msk].reshape(-1, 1), self.devtime[self.msk].reshape(-1, 1)), 3)
        self._Nplus2 = self.predict(np.array(0.8).reshape(-1, 1))[0][0]
        self._Nplus1 = self.predict(np.array(0.67).reshape(-1, 1))[0][0]
        self._N = self.predict(np.array(0.57).reshape(-1, 1))[0][0]
        self._Nminus1 = self.predict(np.array(0.5).reshape(-1, 1))[0][0]
        self._Nminus2 = self.predict(np.array(0.44).reshape(-1, 1))[0][0]

    def model_name(self):
        return __class__.__name__;


def format_devtime(devtime):
    fraction, whole = math.modf(devtime)
    return '{}:{}'.format(
            int(whole),
            str(int(60*fraction)).zfill(2))


def evaluate(file, model):
    Nplus2 = format_devtime(model.Nplus2)
    Nplus1 = format_devtime(model.Nplus1)
    N = format_devtime(model.N)
    Nminus1 = format_devtime(model.Nminus1)
    Nminus2 = format_devtime(model.Nminus2)
    fig, ax = plt.subplots()
    ax.set_title('Lambrecht/Woodhouse', fontsize=10)
    ax.grid(which='both')
    ax.scatter(model.gamma, model.devtime, marker='+', color='blue')
    ax.scatter(0.8, model.Nplus2, marker='.', color='red', label='N+2 = {}'.format(Nplus2))
    ax.scatter(0.67, model.Nplus1, marker='.', color='red', label='N+1 = {}'.format(Nplus1))
    ax.scatter(0.57, model.N, marker='.', color='red', label='N = {}'.format(N))
    ax.scatter(0.5, model.Nminus1, marker='.', color='red', label='N-1 = {}'.format(Nminus1))
    ax.scatter(0.44, model.Nminus2, marker='.', color='red', label='N-2 = {}'.format(Nminus2))
    line = np.linspace(0.3, 0.9, 200)
    ax.plot(line, model._model.predict(line.reshape(-1, 1)), linewidth=1.0, color='blue')
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('gamma')
    ax.set_ylabel('devtime')
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)
    handles, labels = ax.get_legend_handles_labels()
    desc = 'r^2= {} ({})'.format(
        model.r_square,
        model.model_name())
    handles.append(mpatches.Patch(color='none', label=desc))
    plt.legend(handles=handles, fontsize=8)
    file = file.with_suffix('.png')
    plt.savefig(str(file), dpi=300)
    tbl = PrettyTable()
    tbl.field_names = ["Zone development", "development time (21°C)"]
    tbl.add_row(["N+2", Nplus2])
    tbl.add_row(["N+1", Nplus1])
    tbl.add_row(["N", N])
    tbl.add_row(["N-1", Nminus1])
    tbl.add_row(["N-2", Nminus2])
    return tbl


def parse_data(file):
    data = []
    with open(str(file),'rb') as f:
        Lines = f.readlines()
        for line in Lines:
            data.append(json.loads(line))
    data.sort()
    print(data)
    return data


def main(file):
    data = parse_data(file)
    model_theil = TheilSenRegressionModel(data)
    model_ransac = RANSACRegressionModel(data)
    tbl = evaluate(file, model_ransac) if model_ransac.r_square > model_theil.r_square else evaluate(file, model_theil)
    print(tbl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    file = Path(args.file).resolve()
    assert file.exists()
    main(file)
