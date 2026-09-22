"""Import the sibling MORAI expert without ROS startup or a second code copy."""
import importlib
import importlib.util
from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[2]/'MoraiDrive/expert/src/expert/src'
PACKAGE = '_aimdrive_morai_expert'


def expert_module(name):
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, SOURCE/'__init__.py', submodule_search_locations=[str(SOURCE)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    return importlib.import_module(PACKAGE+'.'+name)
