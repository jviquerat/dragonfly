# Generic imports
import sys

# Custom imports
from dragonfly.src.core.train    import train
#from dragonfly.src.core.evaluate import evaluate

def main():

    # Check arguments
    args = sys.argv

    if ("--train" in args):
        json_file = args[args.index("--train")+1]
        train(json_file)

    if ("--eval" in args):
        net_file  = args[args.index("--eval")+1]
        json_file = args[args.index("--eval")+2]
        evaluate(net_file, json_file)
