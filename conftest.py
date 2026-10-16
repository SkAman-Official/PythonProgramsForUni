"""Load independent examples without requiring package installation."""

import importlib.util
import sys
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def project(request: pytest.FixtureRequest) -> ModuleType:
    """Return the source module belonging to a project's test directory."""
    source = request.path.parent.parent / "main.py"
    name = source.parent.name
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
