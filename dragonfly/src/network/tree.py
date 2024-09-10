###############################################
### Trunk class
class trunk:
    def __init__(self):

        # Feel empty struct
        self.arch = [64]
        self.actv = "relu"


###############################################
### Heads class
class heads:
    def __init__(self):

        # Feel empty struct
        self.nb = 1
        self.arch = [[64]]
        self.actv = ["relu"]
        self.final = ["linear"]
