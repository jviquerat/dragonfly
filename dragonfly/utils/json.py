# Generic imports
import json
import types

###############################################
### json parser class
### Used to parse input json files
class json_parser():
    def __init__(self):
        self.pms = None

    def decoder(self, p):
        return types.SimpleNamespace(**p)

    def read(self, filename):
        with open(filename, "r") as f:
            self.pms = json.load(f, object_hook=self.decoder)

        return self.pms
