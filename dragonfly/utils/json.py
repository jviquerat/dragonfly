# Generic imports
import json
import collections

###############################################
### json parser class
### Used to parse input json files
class json_parser():
    def __init__(self):
        pass

    def decoder(self, pdict):
        return collections.namedtuple('X', pdict.keys())(*pdict.values())

    def read(self, filename):
        with open(filename, "r") as f:
            params = json.load(f, object_hook=self.decoder)

        return params
