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


class Model(abc.ABC):
    @abc.abstractmethod
    def _process(self):
        'process data'

    def __init__(self, data):
        self._devtime, self._gamma, _ , _ = zip(*data)
        self._devtime = np.array(self._devtime)
        self._gamma = np.array(self._gamma)
        self._process()

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

    def predict(self, x):
        return self._model.predict(x)


'metric based on: Lambrecht/Woodhouse, Way Beyond Monochrome'
class TheilSenRegressionModel(Model):
    def __init__(self, data):
        super(TheilSenRegressionModel, self).__init__(data)

    def _process(self):
        self._model = TheilSenRegressor().fit(self.gamma.reshape(-1, 1), self.devtime.reshape(-1, 1))
        self._r2 = round(self._model.score(self.gamma.reshape(-1, 1), self.devtime.reshape(-1, 1)), 3)
        self._Nplus2 = self.predict(np.array(0.8).reshape(-1, 1))[0]
        self._Nplus1 = self.predict(np.array(0.67).reshape(-1, 1))[0]
        self._N = self.predict(np.array(0.57).reshape(-1, 1))[0]
        self._Nminus1 = self.predict(np.array(0.5).reshape(-1, 1))[0]
        self._Nminus2 = self.predict(np.array(0.44).reshape(-1, 1))[0]


class RANSACRegressionModel(Model):
    def __init__(self, data):
        super(RANSACRegressionModel, self).__init__(data)

    def _process(self):
        self._model = RANSACRegressor().fit(self.gamma.reshape(-1, 1), self.devtime.reshape(-1, 1))
        self._r2 = round(self._model.score(self.gamma.reshape(-1, 1), self.devtime.reshape(-1, 1)), 3)
        self._Nplus2 = self.predict(np.array(0.8).reshape(-1, 1))[0][0]
        self._Nplus1 = self.predict(np.array(0.67).reshape(-1, 1))[0][0]
        self._N = self.predict(np.array(0.57).reshape(-1, 1))[0][0]
        self._Nminus1 = self.predict(np.array(0.5).reshape(-1, 1))[0][0]
        self._Nminus2 = self.predict(np.array(0.44).reshape(-1, 1))[0][0]


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
    file = file.with_suffix('.png')
    plt.savefig(str(file), dpi=300)
    tbl = PrettyTable()
    tbl.field_names = ["Zone development", "development time (18°C)"]
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
    model = TheilSenRegressionModel(data)
    #model = RANSACRegressionModel(data)
    tbl = evaluate(file, model)
    print(tbl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    file = Path(args.file).resolve()
    assert file.exists()
    main(file)
