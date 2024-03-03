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
import ast
import configparser
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sys

from functools import reduce
from math import log10
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
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
    _min_density = 0.17
    _max_density = 1.37
    _msk = np.array([])

    @abc.abstractmethod
    def _process(self):
        'process data'

    def __init__(self, data):
        self._exposure, self._density = zip(*data)
        self._exposure = np.array(self._exposure)
        self._density = np.array(self._density)
        self._msk = outliers_mask(self._exposure, self._density)
        self._process()

    @abc.abstractmethod
    def model_name(self):
        'model name'

    @property
    def exposure(self):
        return self._exposure

    @property
    def density(self):
        return self._density

    @property
    def min_density(self):
        return self._min_density

    @property
    def max_density(self):
        return self._max_density

    @property
    def exposure_at_min_density(self):
        return self._exposure_at_min_density

    @property
    def exposure_at_max_density(self):
        return self._exposure_at_max_density

    @property
    def null_point(self):
        return self._null_point

    @property
    def gamma(self):
        return self._gamma

    @property
    def r_square(self):
        return self._r2

    @property
    def lower_offset(self):
        return 0.1

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
        self._model = TheilSenRegressor().fit(self.exposure[self.msk].reshape(-1, 1), self.density[self.msk].reshape(-1, 1))
        self._r2 = round(self._model.score(self.exposure[self.msk].reshape(-1, 1), self.density[self.msk].reshape(-1, 1)), 3)
        self._gamma = self._model.coef_[0]
        # exposure exposure_at_mindensity at min density = 0.17
        self._exposure_at_min_density = round((0.17 - self._model.intercept_)/self._gamma, 2)
        self._exposure_at_max_density = round((1.37 - self._model.intercept_)/self._gamma, 2)
        self._null_point = round((0 - self._model.intercept_)/self._gamma, 2)

    def model_name(self):
        return __class__.__name__;


class RANSACRegressionModel(Model):
    def __init__(self, data):
        super(RANSACRegressionModel, self).__init__(data)

    def _process(self):
        self._model = RANSACRegressor().fit(self.exposure[self.msk].reshape(-1, 1), self.density[self.msk].reshape(-1, 1))
        self._r2 = round(self._model.score(self.exposure[self.msk].reshape(-1, 1), self.density[self.msk].reshape(-1, 1)), 3)
        self._gamma = self._model.estimator_.coef_[0][0]
        # exposure exposure_at_mindensity at min density = 0.17
        self._exposure_at_min_density = round((0.17 - self._model.estimator_.intercept_[0])/self._gamma, 2)
        self._exposure_at_max_density = round((1.37 - self._model.estimator_.intercept_[0])/self._gamma, 2)
        self._null_point = round((0 - self._model.estimator_.intercept_[0])/self._gamma, 2)

    def model_name(self):
        return __class__.__name__;


def relative_exposure(density):
    return 10 * log10(2) - density


def average_density(fog, data):
    data = [(exposure, ast.literal_eval(density)) for (exposure, density) in data]
    data = data[::-1]
    return [(round(relative_exposure(float(exposure)), 3), round(round(reduce(lambda a, b: float(a) + float(b), density) / len(density), 3) - fog, 3)) for exposure, density in data]


def parse_data(file_p):
    config = configparser.ConfigParser()
    config.read(str(file_p))
    meta = config['META']
    data = config.items('DATA')
    data = average_density(float(meta['fog']), data)
    return meta, data


def zone_from_density(density):
    return 10 - ( (10 * log10(2) - density) / log10(2))


def evaluate(file_p, model, meta):
    fig, ax = plt.subplots()
    ax.set_title('{}: {}@{} {}[{}] {}/{} {}'.format(
        'Lambrecht/Woodhouse',
        meta['film'],
        meta['ISO'],
        meta['developer'],
        meta['dilution'],
        meta['time'],
        meta['temperature'],
        meta['aggitation']),
        fontsize=10)
    ax.grid(which='both')
    ax.scatter(model.exposure[model.msk], model.density[model.msk], marker='+', color='limegreen')
    ax.scatter(model.exposure[np.invert(model.msk)], model.density[np.invert(model.msk)], marker='+', color='black')
    ax.scatter(model.exposure_at_min_density, model.min_density, marker='.', color='red', label='exposure(Dmin)={}'.format(model.exposure_at_min_density))
    ax.scatter(model.exposure_at_max_density, model.max_density, marker='.', color='red', label='exposure(Dmax)={}'.format(model.exposure_at_max_density))
    line = np.linspace(model.null_point, max(model.exposure_at_max_density, model.exposure[-1]), 200)
    ax.plot(line, model._model.predict(line.reshape(-1, 1)), linewidth=1.0, label='linear regression', color='blue')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('exposure')
    ax.set_ylabel('density')
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)
    handles, labels = ax.get_legend_handles_labels()
    desc = 'gamma = {}\nr^2= {} ({})'.format(
        round(model.gamma, 2),
        model.r_square,
        model.model_name())
    handles.append(mpatches.Patch(color='none', label=desc))
    plt.legend(handles=handles, fontsize=8)
    file_p = file_p.with_suffix('.png')
    plt.savefig(str(file_p), dpi=300)
    tm = meta['time'].split(':')
    tm = round(float(tm[0]) + int(tm[1])/60, 1)
    return (tm, round(model.gamma, 2), round(model.exposure_at_min_density, 2), round(zone_from_density(model.exposure_at_min_density), 1))


def serialize(result):
    print(json.dumps(result))


def main(file):
    meta, data = parse_data(file)
    model_theil = TheilSenRegressionModel(data)
    model_ransac = RANSACRegressionModel(data)
    result = evaluate(file, model_ransac, meta) if model_ransac.r_square > model_theil.r_square else evaluate(file, model_theil, meta)
    serialize(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    file = Path(args.file).resolve()
    assert file.exists()
    main(file)
