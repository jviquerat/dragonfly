# Generic imports
import sys

# Custom imports
from dragonfly.src.core.train    import *
from dragonfly.src.core.evaluate import *
from dragonfly.src.utils.prints  import *

def error():
    new_line()
    errr("""Command line error. Possible behaviors:
    dgf --train <json_file>
    dgf --eval -net    <net_file>
               -json   <json_file>
               -steps  <n_steps> (optional)
               -warmup <n_warmup> <a_warmup> (optional, requires -steps option)""")

def main():

    # Printings
    disclaimer()

    # Check arguments
    args = sys.argv

    # Training mode
    if ("--train" in args):
        new_line()
        liner_simple()
        bold('Training mode')

        json_file = args[args.index("--train")+1]
        train(json_file)
        return

    # Evaluation mode
    if ("--eval" in args):
        new_line()
        liner_simple()
        bold('Evaluation mode')

        if ("-net" not in args): error()
        net_file  = args[args.index("-net")+1]

        if ("-json" not in args): error()
        json_file = args[args.index("-json")+1]

        n_steps = 0
        if ("-steps" in args):
            n_steps = int(args[args.index("-steps")+1])

        n_warmup = 0
        a_warmup = None
        if ("-warmup" in args):
            if ("-steps" not in args):
                error()
            n_warmup = int(args[args.index("-warmup")+1])
            a_warmup = args[args.index("-warmup")+2]

        evaluate(net_file, json_file, n_steps, n_warmup, a_warmup)
        return

if __name__ == "__main__":
    main()
