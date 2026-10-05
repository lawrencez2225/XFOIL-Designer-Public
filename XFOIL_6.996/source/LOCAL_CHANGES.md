# XFOIL source and local changes

`xfoil6.996.tgz` is the upstream source archive. It was downloaded again
from MIT on 2026-10-05 and its SHA-256 exactly matches the archive here:
`0feb71b58070d8514d830fcb0b609a10a11399d362493d7b9bbfb7343dff3f23`.

The extracted `Xfoil/` tree includes two substantive local changes:

- `src/xoper.f`: the `CSAV` command saves Cp for converged points using
  sequence-index filenames; sequence plotting restores the foreground color.
  These changes were present in the private project on 2026-07-12 and
  2026-07-14. The public copy adds a dated modification notice only.
- `plotlib/config.make`: uses double precision, `DBL_ARGS`, and the XQuartz
  include directory. This build configuration was present on 2026-07-12;
  the public copy adds a dated modification notice only.

Some upstream examples have normalized line endings. The GPL and GNU Library
GPL notices in the source files remain intact. The public copy adds full
license texts and these descriptions; it does not change executable Fortran
statements, Python calculations, or the previously tested native artifacts.

The binaries are local Apple Silicon builds, not official MIT macOS binaries.
The original exact compiler invocation was not retained. The included build
configuration describes double precision, and the source below documents a
fresh build; bit-for-bit reproduction of the historical binaries is not claimed.

## Building from the supplied source

Use a copy of `Xfoil/` to keep the shipped binaries intact. A GNU Fortran
compiler, C compiler, make, and X11 development headers/libraries are needed.
On Apple Silicon the compiler and X11 libraries must target arm64. XQuartz
provides headers under `/opt/X11/include` and libraries under `/opt/X11/lib`.

From that source copy, the following commands use the existing makefiles:

```sh
make -C plotlib -f Makefile.all lib FC=gfortran CC=cc \
  FFLAGS="-O2 -fdefault-real-8 -std=legacy -fallow-argument-mismatch" \
  CFLAGS="-O2 -DUNDERSCORE -DDBL_ARGS" \
  INCDIR="-I/opt/X11/include"
mkdir -p local-bin
make -C bin -f Makefile_gfortran all FC=gfortran CC=cc \
  FFLAGS="-O -fdefault-real-8 -std=legacy -fallow-argument-mismatch" \
  FFLOPT="-O -fdefault-real-8 -std=legacy -fallow-argument-mismatch" \
  PLTLIB="-L/opt/X11/lib -lX11" BINDIR=../local-bin
```

These are source-build instructions, not a record of the original build.
Rerun the native integration tests with a rebuilt solver before replacing
the supplied binaries. Source and binaries retain their separate licenses
as described in `../LICENSE-README.md`.
