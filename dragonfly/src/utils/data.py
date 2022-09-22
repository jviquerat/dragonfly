# Generic imports
import numpy as np

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
        self.ep   = np.zeros((       n_stp          ), dtype=int)
        self.stp  = np.zeros((       n_stp          ), dtype=int)
        self.data = np.zeros((n_avg, n_stp, n_fields), dtype=float)

    def store(self, filename, run):

        f = np.loadtxt(filename)
        self.stp = f[:self.n_stp, 0]
        self.ep  = f[:self.n_stp, 1]
        for field in range(self.n_fields):
            self.data[run,:,field] = f[:self.n_stp,field+2]

    def average(self, filename):

        array = np.vstack(self.stp)
        array = np.hstack((array, np.vstack(self.ep)))

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
