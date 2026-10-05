# XFOIL, Xplot11 and bundled runtime licenses

This directory contains independent third-party works. The root Apache-2.0
license applies to the project's own Python code, not these components.

## XFOIL

The Fortran source headers in `source/Xfoil/src/xfoil.f` and `xoper.f`
explicitly grant **GPL version 2 or (at your option) any later version**.
The GPL-2.0 text is `COPYING`; the GPL-3.0 text is `COPYING3`.
For this distribution, the combined solver executables use the GPL-3.0
option compatible with the included GPL-3.0 GCC runtime libraries.
The separate Xplot11 source files keep their own original license notices.

`bin/{xfoil,pplot,pxplot}` are locally built Apple Silicon macOS executables
based on Mark Drela's XFOIL 6.996 and Harold Youngren's plotting library.
They are not official MIT macOS binaries. Their source, including local
changes and build files, is in `source/Xfoil/`; the unchanged upstream archive
is `source/xfoil6.996.tgz`. Local changes and source-build instructions are
recorded in `source/LOCAL_CHANGES.md`.

The current binaries have debugging symbols removed and fresh ad-hoc
signatures. Calculating code and data were preserved; checksums and numerical
validation are recorded in `runtime/packaging.json`.

## Xplot11

The source headers in `source/Xfoil/plotlib/plt_base.f`, `xwin11/Xwin2.c`,
and other plotting-library files state **GNU Library General Public License
version 2 or later**. The full version-2 text is provided at
`source/Xfoil/plotlib/COPYING.LIB`. GPL conversion of the linked copy is
permitted by section 3 of that license; this does not replace the original
notices or relicense the separate source files under Apache-2.0.

## Runtime libraries and X11 data

Library identities, upstream package URLs, hashes, and licenses are recorded
in `runtime/conda-meta/`. These records identify provenance; they are not a
Conda environment that can be activated or used to reproduce all installed
files. Complete copyright/license texts are under `runtime/share/licenses/`.

GCC libraries: GPL-3.0-only WITH GCC-exception-3.1. Full GPL-3.0 and exception
texts are included. Exact source-download directions and the matching Conda
build scripts and patches are supplied in `runtime/source/README.md`.
X11 libraries and data: upstream MIT-style notices are preserved in each
library's COPYING and in the individual data files.

Fourteen X11 Compose files were adapted from unusable absolute build-prefix
includes to the relative paths already used by the package's `compose.dir`.
Their copyright notices were kept. libX11 resource strings were relocated
using Conda's binary-prefix handling; the library code was not changed.

## Separate application

The Python application launches XFOIL as a separate process and exchanges
command and result files. Its Apache-2.0 license does not replace any of
the licenses above. See `../THIRD_PARTY_NOTICES.md` for the component inventory.
