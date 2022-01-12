# Generic imports
import numpy as np

###############################################
### Report buffer, used to store learning metrics
class report:
    def __init__(self, names):

        self.names = names
        self.reset()

    # Reset
    def reset(self):

        self.data = {}
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
    def write(self, filename):

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
