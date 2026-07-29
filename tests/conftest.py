import glob
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))


@pytest.fixture(scope="session")
def upstream():
    pattern = os.path.join(
        sys.prefix, "lib", "python*", "site-packages", "pymeshfix", "_meshfix*.so"
    )
    matches = glob.glob(pattern)
    if not matches:
        pytest.skip("real upstream pymeshfix extension is not installed")
    spec = importlib.util.spec_from_file_location("_meshfix", matches[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
