# Custom imports
from dragonfly.src.utils.data import *
from dragonfly.src.plot.plot  import *

# Average existing runs
def average(args):

    # Count arguments
    n_args = len(args)

    # Get length of file
    with open(args[0], 'r') as f:
        n_lines = sum(1 for line in f)

    # Intialize averager
    averager = data_avg(2, n_lines, n_args)

    # Run
    for run in range(n_args):
        filename = args[run]
        averager.store(filename, run)

    # Write to file
    filename = 'avg.dat'
    data = averager.average(filename)

    # Plot
    filename = 'avg'
    plot_avg(data, filename)
