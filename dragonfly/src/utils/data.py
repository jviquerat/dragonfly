# Generic imports
import numpy as np

# Custom imports
from dragonfly.src.utils.ema import *

###############################################
### Data averager class
### Used to compute avg+/-std of drl-related fields
### n_fields : nb of fields to store/average
### n_stp    : nb of steps per run
### n_avg    : nb of runs to average
class data_avg():
    def __init__(self, n_fields, n_stp, n_avg):

        self.n_stp    = n_stp
        self.n_fields = n_fields
        self.n_avg    = n_avg
        self.stp  = np.zeros((       n_stp          ), dtype=int)
        self.ep   = np.zeros((n_avg, n_stp          ), dtype=int)
        self.data = np.zeros((n_avg, n_stp, n_fields), dtype=float)

    def store(self, filename, run):

        f = np.loadtxt(filename)
        self.stp       = f[:self.n_stp, 0]
        self.ep[run,:] = f[:self.n_stp, 1]
        for field in range(self.n_fields):
            self.data[run,:,field] = f[:self.n_stp,field+2]

    def smooth(self):

        for r in range(self.n_avg):
            for f in range(self.n_fields):
                avg = ema(0.01, 50)
                for i in range(self.n_stp):
                    avg.add(self.data[r,i,f])
                    self.data[r,i,f] = avg.avg()

    def average(self, filename):

        array = np.vstack(self.stp)
        avg   = np.mean(self.ep[:,:], axis=0)
        array = np.hstack((array, np.vstack(avg)))

        for field in range(self.n_fields):
            avg   = np.mean(self.data[:,:,field], axis=0)
            std   = np.std (self.data[:,:,field], axis=0)
            p     = avg + std
            m     = avg - std
            array = np.hstack((array,np.vstack(avg)))
            array = np.hstack((array,np.vstack(p)))
            array = np.hstack((array,np.vstack(m)))

        np.savetxt(filename, array, fmt='%.5e')
        return array
