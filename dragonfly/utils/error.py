# Custom imports
from dragonfly.utils.prints import *

###############################################
### A set of functions to print errors and warnings

### Error
def error(module, function, text):
    liner()
    errr("Dragonfly error")
    spacer()
    print("Module "+str(module)+", function "+str(function))
    spacer()
    print(text)
    exit(1)

### Warning
def warning(module, function, text):
    liner()
    warn("Dragonfly warning")
    spacer()
    print("Module "+str(module)+", function "+str(function))
    spacer()
    print(text)
    new_line()

