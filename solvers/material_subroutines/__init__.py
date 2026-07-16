"""mlhp user-material subroutines written in C and compiled at runtime with cffi.

Each module holds one material law as a C source string plus a ``build()`` that
compiles it through :func:`build_subroutine` and returns the loaded library
exposing its callbacks as integer function pointers for
``mlhp.constitutiveEquation`` (and ``mlhp.meshFunctionStrainUpdate`` where the
law carries history).
"""

import importlib
import sys
from pathlib import Path

import cffi


def build_subroutine(name, source, symbols, tmpdir=None):
    """Compile a C subroutine with cffi and return the loaded library.

    Args:
        name: extension module name, e.g. ``"_j2_subroutine"``.
        source: the C source exposing each symbol as
            ``const unsigned long long <symbol> = (unsigned long long)&fn;``.
        symbols: the exported address symbols, e.g. ``["material_address"]``.
        tmpdir: build directory (defaults to ``_build`` next to this package).
    """
    tmpdir = Path(tmpdir) if tmpdir else Path(__file__).parent / "_build"
    tmpdir.mkdir(parents=True, exist_ok=True)

    ffi = cffi.FFI()
    ffi.cdef(f"extern const unsigned long long {', '.join(symbols)};")
    ffi.set_source(name, source)
    ffi.compile(tmpdir=str(tmpdir))

    if str(tmpdir) not in sys.path:
        sys.path.insert(0, str(tmpdir))
    return importlib.import_module(name).lib
