# PyGTO

Python tools for constructing, optimizing, reducing, and contracting Gaussian-type
orbital (GTO) basis sets.

<!-- TODO: Add PyPI, supported-Python, test-status, and documentation badges. -->

> **Development status:** PyGTO is under active development and is being prepared for
> its first PyPI release. APIs may still change before version 1.0.

PyGTO provides basis-set representations that are independent of any particular
electronic-structure program. Optimization and workflow classes accept ordinary Python
callables, allowing them to work with different computational backends. PySCF is the
reference backend and is supported through optional helper functions.

## Features

- Represent optimizable primitive basis sets with `BasisSpec` and `Channel` objects.
- Represent contracted basis sets with `ContractedBasis` and `ContractedChannel`.
- Read named, NWChem-format, and PySCF-format basis data.
- Parameterize exponents independently or with constraints (e.g., an even-tempered form).
- Optimize exponents with Nelder-Mead, BFGS, or the recommended staged optimizer.
- Reduce basis size while retaining a specified target accuracy.
- Optimize auxiliary and material-constrained atomic-orbital basis sets.
- Generate atomic-natural-orbital (ANO) contractions and partial decontractions.
- Use PySCF when available without requiring it for the core basis representations.

## Installation

PyGTO has not yet had its first PyPI release. Once published, the planned installation
commands are:

```bash
# Core package
python -m pip install pygto

# Core package with the PySCF backend
python -m pip install "pygto[pyscf]"

# Optional Basis Set Exchange Python client
python -m pip install "pygto[bse]"

# All optional runtime dependencies
python -m pip install "pygto[all]"
```

For development, after packaging metadata has been added:

```bash
git clone https://github.com/ye-chem-group/pygto.git
cd pygto
python -m pip install -e ".[pyscf,test]"
```

The planned core dependencies are NumPy, SciPy, and h5py. PySCF is optional but is
required for the electronic-structure calculations used in most workflow examples.

<!-- TODO: Replace “planned” installation text after the first PyPI release. -->

## Quick start

### Construct a basis specification

The central optimizable representation is `BasisSpec`, which stores primitive exponents
in angular-momentum channels. For example, the following code generates a 4s4p3d
even-tempered basis for carbon:

```python
from pygto.basis import BasisSpec

etb_params = [
    # l, nprim, amin, beta
    (0, 4, 0.1, 3.5),
    (1, 4, 0.1, 3.5),
    (2, 3, 0.2, 3.5),
]

spec = BasisSpec.init_from_etb_params(etb_params, atm="C")
spec.log_note(spec.structure)
spec.dump_basis()
```

A `BasisSpec` can also be initialized from a standard named basis:

```python
from pygto.basis import BasisSpec

spec = BasisSpec.init_from_basis("cc-pvdz", "C")
spec.dump_basis()
```

When PySCF is installed, PyGTO uses its basis loader. Otherwise, named basis data are
obtained through Basis Set Exchange, using either its optional Python package or the BSE
REST API.

### Optimize a basis with PySCF

`ScheduledOptimizer` is the recommended default optimizer. Its default schedule performs
a derivative-free Nelder-Mead optimization followed by BFGS refinement:

```python
from pyscf import scf

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import ScheduledOptimizer

atm = "C"
spin = 2
etb_params = [
    #l, nprim, amin, beta
    (0, 9, 0.1, 3.5),
    (1, 4, 0.1, 3.5),
]
spec = BasisSpec.init_from_etb_params(
    etb_params, atm=atm).convert_to("full")

cost_func = lib.pyscf_helper.get_cost_func(
    atm, scf.ROHF, mol_settings={"spin": spin},
)

opt = ScheduledOptimizer(spec, cost_func)
cost, opt_spec = opt.kernel()

opt_spec.log_note('ROHF energy: %.10f' % cost)
opt_spec.dump_basis()
```

## Examples

Curated examples are organized by topic:

- [Basis construction and inspection](https://github.com/ye-chem-group/pygto/tree/main/examples/basis)
- [Individual and scheduled optimizers](https://github.com/ye-chem-group/pygto/tree/main/examples/optimizer)
- [Basis-design workflows](https://github.com/ye-chem-group/pygto/tree/main/examples/workflow)


## Documentation

To be added.


## Contributing

Bug reports, feature requests, and pull requests are welcome through the
[GitHub repository](https://github.com/ye-chem-group/pygto).


## Citation

If PyGTO contributes to published work, please cite the software and the associated
methodology papers where appropriate.

To be added.


## License

PyGTO is distributed under the BSD 3-Clause License. See
[`LICENSE`](https://github.com/ye-chem-group/pygto/blob/main/LICENSE) for details.

## Contact

PyGTO is developed by the Ye Research Group at the University of Maryland.
