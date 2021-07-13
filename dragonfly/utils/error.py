###############################################
### Header for errors and warnings
def header():
    print("\n")
    print("######################")

### Error
def error(module, function, text):
    header()
    print("### Dragonfly error")
    print("### Module "+str(module)+", function "+str(function))
    print("### "+text)
    exit(1)

### Warning
def warning(module, function, text):
    header()
    print("### Dragonfly warning")
    print("### Module "+str(module)+", function "+str(function))
    print("### "+text)
    print("\n")

