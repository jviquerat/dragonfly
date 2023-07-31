# Generic imports
import os
import subprocess

# Custom improts
from dragonfly.src.env.mpi import *

###############################################
### A set of functions to format printings

### Specific colors for printings
wrn_clr = '\033[93m'
err_clr = '\033[91m'
end_clr = '\033[0m'
bld_clr = '\033[1m'

### New line
def new_line():
    if (mpi.rank == 0):
        print("")

### Header
def header():
    if (mpi.rank == 0):
        print("#################################")

### Liner
def liner():
    if (mpi.rank == 0):
        new_line()
        print("###", end=" ")

### Liner with no newline
def liner_simple():
    if (mpi.rank == 0):
        print("###", end=" ")

### Spacer
def spacer():
    if (mpi.rank == 0):
        print("#", end=" ")

### Dragonfly disclaimer
def disclaimer():
    if (mpi.rank == 0):
        header()
        liner_simple()
        bold("Dragonfly, a DRL library")
        liner_simple()
        git_short_hash()
        header()

### Print with warning color
def warn(text):
    if (mpi.rank == 0):
        print(wrn_clr + text + end_clr)

### Print with error color
def errr(text):
    if (mpi.rank == 0):
        print(err_clr + text + end_clr)

### Print with bold text
def bold(text):
    if (mpi.rank == 0):
        print(bld_clr + text + end_clr)

### Print git revision
def git_short_hash() -> str:
    if (mpi.rank == 0):
        try:
            process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                                       shell=False,
                                       stdout=subprocess.PIPE)
            hash = process.communicate()[0].decode('ascii').strip()
            print("Revision "+str(hash))
        except Exception as e:
            pass
