# Generic imports
import numpy as np

###############################################
### Data averager class
### Used to compute avg+/-std of drl-related fields
### n_fields : nb of fields to store/average
### n_ep     : nb of episodes per run
### n_avg    : nb of runs to average
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
        self.ep = f[:self.n_ep, 0]
        for field in range(self.n_fields):
            self.data[run,:,field] = f[:self.n_ep,field+1]

    def average(self, filename):
        array     = np.vstack(self.ep)
        for field in range(self.n_fields):
            avg   = np.mean(self.data[:,:,field], axis=0)
            std   = np.std (self.data[:,:,field], axis=0)
            p     = avg + std
            m     = avg - std
            array = np.hstack((array,np.vstack(avg)))
            array = np.hstack((array,np.vstack(p)))
            array = np.hstack((array,np.vstack(m)))

        np.savetxt(filename, array, fmt='%.5e')
