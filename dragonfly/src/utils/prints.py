###############################################
### A set of functions to format printings

### Specific colors for printings
wrn_clr = '\033[93m'
err_clr = '\033[91m'
end_clr = '\033[0m'
bld_clr = '\033[1m'

### New line
def new_line():
    print("")

### Header
def header():
    print("#################################")

### Liner
def liner():
    new_line()
    print("###", end=" ")

### Liner with no newline
def liner_simple():
    print("###", end=" ")

### Spacer
def spacer():
    print("#", end=" ")

### Dragonfly disclaimer
def disclaimer():
    header()
    liner_simple()
    bold("Dragonfly, a DRL library")
    header()

### Print with warning color
def warn(text):
    print(wrn_clr + text + end_clr)

### Print with error color
def errr(text):
    print(err_clr + text + end_clr)

### Print with bold text
def bold(text):
    print(bld_clr + text + end_clr)
