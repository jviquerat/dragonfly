# Generic imports
import math
import numpy      as np
import tensorflow as tf

###############################################
### Parallel buffer class, used to temporarily
### store data from parallel environments
class pbuff:
    def __init__(self, n_cpu, dim):

        self.n_cpu = n_cpu
        self.dim   = dim
        self.reset()

    def reset(self):

        self.buff = [np.array([]) for _ in range(self.n_cpu)]

    def append(self, vec):

        for cpu in range(self.n_cpu):
            self.buff[cpu] = np.append(self.buff[cpu], vec[cpu])

    def length(self):

        return int(len(self.buff[0])/self.dim)

    def serialize(self):

        arr = np.array([])
        for cpu in range(self.n_cpu):
            arr = np.append(arr, self.buff[cpu])

        return np.reshape(arr, (-1,self.dim))

###############################################
### Local parallel buffer class, used to store
### data between two updates of the agent
class buff:
    def __init__(self, n_cpu, names, dims):

        self.n_cpu = n_cpu
        self.names = names
        self.dims  = dims
        self.reset()

    def reset(self):

        self.data = {}
        for name, dim in zip(self.names, self.dims):
            self.data[name] = pbuff(self.n_cpu, dim)

    def store(self, names, fields):

        for name, field in zip(names, fields):
            self.data[name].append(field)

    def size(self):

        return self.length()*self.n_cpu

    def length(self):

        return self.data[self.names[0]].length()

    def serialize(self, names):

        return {name : self.data[name].serialize() for name in names}

###############################################
### Global parallel buffer class, used to store
### all data since the beginning of learning
class gbuff:
    def __init__(self, names, dims):

        self.names = names
        self.dims  = dims
        self.reset()

    def reset(self):

        self.data = {}
        for name, dim in zip(self.names, self.dims):
            self.data[name] = np.empty([0,dim])

    def store(self, names, fields):

        for name, field in zip(names, fields):
            self.data[name] = np.append(self.data[name], field, axis=0)

    def length(self):

        return int(self.data[self.names[0]].shape[0])

    def get_buffers(self, names, size):

        # Start/end indices
        end    = self.length()
        start  = max(0,end - size)
        size   = end - start

        # Randomize batch
        sample = np.arange(start, end)
        np.random.shuffle(sample)

        # Return shuffled fields
        out = {}
        for name in names:
            tmp = [self.data[name][i] for i in sample]
            tmp = tf.cast(tmp, tf.float32)
            dim = self.data[name].shape[1]
            if (dim == 1):
                tmp = tf.reshape(tmp, [size])
            else:
                tmp = tf.reshape(tmp, [size, dim])
            out[name] = tmp

        return {name : out[name] for name in names}
