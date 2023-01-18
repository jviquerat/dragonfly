# Generic imports
import numpy as np

###############################################
### Exponential moving average class
class ema:
    def __init__(self, alpha, n_layers=1):
        self.alpha    = alpha
        self.n_layers = n_layers
        self.y        = np.zeros(n_layers)

    ### Add a value to the buffer
    def add(self, value):
        v = value
        for i in range(self.n_layers):
            if (i>0): v = self.y[i-1]
            self.y[i] = self.alpha*v + (1.0-self.alpha)*self.y[i]

    ### Add a chunk of data to the buffer
    def add_buffer(self, array):
        for i in range(len(array)):
            self.add(array[i])

    ### Return average
    def avg(self):
        return self.y[-1]

###############################################
### Report buffer, used to store learning metrics
# frequency : writing frequency (in number of calls to write() )
# names     : dict names for report
class report:
    def __init__(self, frequency, names):

        self.frequency = frequency
        self.names     = names
        self.reset()

    # Reset
    def reset(self):

        self.count = 0
        self.data  = {}
        for name in self.names:
            self.data[name] = []

    # Append data to the report
    def append(self, name, value):

        self.data[name].append(value)

    # Get data from the report
    def get(self, name):

        return self.data[name]

    # Return an average of n last values of given field
    def avg(self, name, n):

        return np.mean(self.data[name][-n:])

    # Write report
    def write(self, filename, force=False):

        self.count += 1
        if ((self.count%self.frequency == 0) or (force == True)):

            # Generate array to save
            array = np.array([])
            for name in self.names:
                tmp = np.array(self.data[name],dtype=float)
                if array.size: array = np.vstack((array, tmp))
                else:          array = tmp

            array = np.transpose(array)
            array = np.nan_to_num(array, nan=0.0)

            # Save as a csv file
            np.savetxt(filename, array, fmt='%.5e')

            # Reset
            self.count = 0
