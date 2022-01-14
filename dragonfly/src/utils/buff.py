# Generic imports
import math
import numpy      as np
import tensorflow as tf

# Custom imports
from dragonfly.src.utils.error import *

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
### Ring buffer class, used as element of gbuff
class rbuff:
    def __init__(self, size, dim):

        self.size  = size
        self.dim   = dim
        self.reset()

    def reset(self):

        self.i    = 0 # Filling index
        self.n    = 0 # Global  index
        self.full = False
        self.buff = np.zeros([self.size,self.dim])

    def length(self):

        return self.n

    def store(self, field):

        for j in range(len(field)):
            self.buff[self.i,:] = field[j,:]
            self.i             += 1

            # Handle filling and global index
            if (self.i == self.size): self.i    = 0
            if (not self.full):       self.n   += 1
            if (self.n == self.size): self.full = True

    def get_dim(self):

        return self.dim

    def get_indices(self, size):

        end = self.i
        if (    self.full): start = end-size
        if (not self.full): start = max(0, end-size)

        return start, end

    def get(self, idx):

        return [self.buff[i] for i in idx]

###############################################
### Global parallel buffer class, used to store
### all data since the beginning of learning
class gbuff:
    def __init__(self, size, names, dims):

        self.size  = size
        self.names = names
        self.dims  = dims
        self.reset()

    def reset(self):

        self.data = {}
        for name, dim in zip(self.names, self.dims):
            self.data[name] = rbuff(self.size, dim)

    def store(self, names, fields):

        for name, field in zip(names, fields):
            self.data[name].store(field)

    def length(self):

        return self.data[self.names[0]].length()

    def get_buffers(self, names, size, shuffle=True):

        if (size > self.size):
            error("gbuff",
                  "get_buffers",
                  "Size too large for buffer")

        # Start/end indices
        start, end = self.data[self.names[0]].get_indices(size)
        s          = end-start

        # Randomized indices
        smp = np.arange(start, end)
        if (shuffle): np.random.shuffle(smp)

        # Return shuffled fields
        out = {}
        for name in names:
            tmp = self.data[name].get(smp)
            tmp = tf.cast(tmp, tf.float32)
            dim = self.data[name].get_dim()
            if (dim == 1):
                tmp = tf.reshape(tmp, [s])
            else:
                tmp = tf.reshape(tmp, [s, dim])
            out[name] = tmp

        return {name : out[name] for name in names}
