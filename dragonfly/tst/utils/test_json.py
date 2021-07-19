# Generic imports
import pytest

# Custom imports
from dragonfly.tst.tst        import *
from dragonfly.src.utils.json import *

###############################################
### Test json reader
def test_json_reader():

    # Initial space
    print("")

    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read("dragonfly/tst/utils/json_file.json")

    print("Test contents")
    assert(reader.pms.env_name == "test_env")
    assert(reader.pms.policy.network.type == "fc")
    print("")

    print("Test contents extraction")
    pms_extract = reader.pms.policy.network
    assert(pms_extract.trunk.arch[0] == 16)
    print("")

    print("Test adding content")
    with pytest.raises(Exception):
        assert(pms_extract.test == 1)
    pms_extract.test = 1
    assert(pms_extract.test == 1)
    print("")
