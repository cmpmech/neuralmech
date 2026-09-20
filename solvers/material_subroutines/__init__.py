"""mlhp user-material subroutines written in C and compiled at runtime with cffi.

Each module holds one material law as a C source string plus a `build()` returning
the loaded library, whose callbacks are handed to `mlhp.constitutiveEquation` (and
`mlhp.meshFunctionStrainUpdate` where the law carries history) as integer function
pointers.
"""

import importlib
import sys
from pathlib import Path

import cffi


def build_subroutine(name, source, symbols, tmpdir=None):
    """compile a C subroutine with cffi and return the loaded library.

    Args:
        name: extension module name, e.g. "_j2_subroutine".
        source: C source exposing each symbol as
            `const unsigned long long <symbol> = (unsigned long long)&fn;`.
        symbols: exported address symbols, e.g. ["material_address"].
        tmpdir: build directory; defaults to `_build` next to this package.
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
