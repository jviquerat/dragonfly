# Custom imports
from dragonfly.src.utils.data   import data_avg
from dragonfly.src.plot.plot    import plot_avg
from dragonfly.src.utils.prints import spacer

# Average existing runs
def average(args):

    # Count arguments
    n_args = len(args)

    # Printing
    spacer()
    print("Averaging files:")
    for i in range(len(args)):
        spacer()
        print(args[i])

    # Get ming length over all files
    n_lines = 1000000000000
    for i in range(len(args)):
        with open(args[i], 'r') as f:
            n_lines = min(n_lines, sum(1 for line in f))

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
