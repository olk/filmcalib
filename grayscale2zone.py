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

import argparse
import ast
import configparser
import numpy as np

from functools import reduce
from math import log10
from pathlib import Path
from prettytable import PrettyTable

def compute_zones(data):
    return [(step, density, 10 - (density/log10(2))) for step, density in data]

def normalize_data(data):
    data = [(x, ast.literal_eval(y)) for (x, y) in data]
    return [(x, np.round(reduce(lambda a, b: float(a) + float(b), y) / len(y), 3)) for x, y in data]

def main(file_p):
    config = configparser.ConfigParser()
    config.read(str(file_p))
    data = config.items('DATA')
    data = normalize_data(data)
    data = compute_zones(data)
    tbl = PrettyTable()
    tbl.field_names = ["step", "density", "zone"]
    tbl.align["step"] = "l"
    tbl.align["density"] = "r"
    tbl.align["zone"] = "r"
    for step, density, zone in data:
        tbl.add_row([step, np.round(density, 2), np.round(zone, 2)])
    print(tbl.get_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    file_p = Path(args.file).resolve()
    assert file_p.exists()
    main(file_p)
