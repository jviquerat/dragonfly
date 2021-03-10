# Generic imports
import json
import collections
import numpy as np

###############################################
### json parser class
### Used to parse input json files
class json_parser():
    def __init__(self):
        pass

    def decoder(self, pdict):
        return collections.namedtuple('X', pdict.keys())(*pdict.values())

    def read(self, filename):
        with open(filename, "r") as f:
            params = json.load(f, object_hook=self.decoder)

        return params

###############################################
### Data averager class
### Used to compute avg+/-std of drl-related fields
class data_avg():
    def __init__(self, n_fields, n_ep, n_avg):
        self.n_ep = n_ep
        self.n_fields = n_fields
        self.ep   = np.zeros((        n_ep            ), dtype=int)
        self.data = np.zeros((n_avg,  n_ep,   n_fields), dtype=float)
        self.avg  = np.zeros((        n_ep,   n_fields), dtype=float)
        self.stdp = np.zeros((        n_ep,   n_fields), dtype=float)
        self.stdm = np.zeros((        n_ep,   n_fields), dtype=float)

    def store(self, filename, run):
        f  = np.loadtxt(filename)
        ep = f[:self.n_ep, 0]
        for field in range(self.n_fields):
            self.data[run,:,field] = f[:self.n_ep,field+1]
