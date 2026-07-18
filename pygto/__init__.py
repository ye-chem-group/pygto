from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pygto")
except PackageNotFoundError:
    __version__ = "0+unknown"


from . import lib
from .basis import BasisSpec, ContractedBasis
from .optimizer import ScheduledOptimizer

__all__ = [
    "BasisSpec",
    "ContractedBasis",
    "ScheduledOptimizer",
    "lib",
]
