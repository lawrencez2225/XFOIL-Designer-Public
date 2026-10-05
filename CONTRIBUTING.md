# Contributing

Thanks for considering a contribution. This is a scientific tool, so the bar
for correctness is higher than for a typical app — see
[Scientific conventions](#scientific-conventions) below.

## Ways to help

| Contribution | Notes |
|---|---|
| **Platform ports** | The highest-value work right now. See [Porting to another platform](#porting-to-another-platform). |
| **English translations** | The `docs/` files are Chinese-only. Translating any one of them is genuinely useful. |
| **Bug reports** | Especially solver-recovery and convergence edge cases. Include the failing case. |
| **Validation data** | Measured airfoil data with a citable source and documented test conditions. |
| **Documentation fixes** | Typos, stale paths, unclear parameter descriptions. |

## Development setup

```bash
git lfs install
git clone https://github.com/lawrencez2225/XFOIL-Designer-Public.git
cd XFOIL-Designer
git lfs pull
conda env create -f environment.yml
conda activate xfoil-work
pip install -e ".[dev,surrogate]"
```

Verify Git LFS actually delivered the runtime libraries before anything else:

```bash
file XFOIL_6.996/runtime/lib/libX11.6.dylib
# expect: Mach-O 64-bit dynamically linked shared library arm64
```

## Running the tests

```bash
# Full suite, including tests that drive the real solver:
MPLBACKEND=Agg XFOIL_NATIVE_TESTS=1 python -m unittest discover -s tests -v

# Fast suite, skipping everything that needs the native binaries:
MPLBACKEND=Agg python -m unittest discover -s tests
```

Most tests use the deterministic `fake_solver` fixture in
[tests/helpers.py](tests/helpers.py) and therefore run on any platform. Only
the tests gated behind `XFOIL_NATIVE_TESTS=1` need the bundled arm64 binaries.
Path discovery still requires a compatible XFOIL executable. On Ubuntu,
install one with `sudo apt-get install xfoil`; the shipped macOS executable
cannot run there. Automatic AVL installation runs only on Apple Silicon
macOS. CI checks Python 3.11 and 3.13 on Linux, and runs the actual XFOIL
and AVL integration tests on macOS.

**Please run the suite before opening a pull request, and say in the PR which
of the two commands you ran.**

## Code style

Configured in [pyproject.toml](pyproject.toml) and [setup.cfg](setup.cfg):

```bash
python -m black .        # line length 79
python -m flake8         # E203 ignored, matching Black's slice spacing
```

Install `.[dev]` to use the same pinned tool versions as CI. Both formatting
checks are required to pass. The 79-column limit is enforced, not advisory.
`XFOIL_6.996/`, `avl_runtime/`, `xfoil_runs/`, and `coord_seligFmt/` are
excluded from both tools.

## Scientific conventions

These are load-bearing in this codebase. A pull request that violates them will
be asked to change, even if the code works.

1. **Never call solver output "experimental data."** Only measured data is
   experimental. `examples/reference/README.md` documents the one experimental
   dataset in the repository and explicitly warns against mixing its grit
   series or treating XFOIL output as measurement.
2. **Do not trust convergence flags alone.** A converged solution can still be
   physically wrong. New analysis paths should surface domain evidence — see
   [docs/TRUST_AND_SURROGATE.md](docs/TRUST_AND_SURROGATE.md) for how the
   existing trust layer does this.
3. **Record provenance.** Anything that produces a number a reader will act on
   should carry where it came from. Follow the existing `*_provenance.json`
   pattern.
4. **State modelling assumptions separately from data.** Transition location,
   grit roughness, and geometry differences are assumptions, not measurements.
5. **Report experimental error and software correctness separately.** Passing
   tests does not validate every aerodynamic case.

## Porting to another platform

The blocker is the solver, not the Python. `xfoil`, `pplot`, and `pxplot` are
arm64 Mach-O binaries linked against bundled macOS `.dylib` files, and
`xfoil_mac/runtime.py` sets `DYLD_LIBRARY_PATH` to find them.

A useful port would:

1. Build XFOIL from the Fortran source already shipped in
   `XFOIL_6.996/source/Xfoil/` (it needs a Fortran toolchain and X11).
2. Extend `xfoil_env()` in `xfoil_mac/runtime.py` to set the right loader path
   per platform (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS).
3. Make `discover_app()` fall back to an `xfoil` executable on `PATH` instead
   of hard-failing when the bundled directory is absent.
4. Replace the hard-coded XQuartz handling in `start_xquartz()` — it assumes
   `/Applications/Utilities/XQuartz.app` and shells out to `open -a`.
5. Emit an explicit "this platform is not supported because …" message rather
   than letting dyld or the kernel produce a confusing failure.

Opening an issue to discuss the approach before writing it is welcome.

## Licensing of contributions

This project is licensed under Apache-2.0. By submitting a contribution you
agree that it is your own work and that it may be distributed under those
terms. If your contribution touches third-party material — data, code, or
binaries — say so in the pull request so
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) can be updated.

Note the precedent in that file: the `coord_seligFmt/` airfoil database is
copyrighted by UIUC with no redistribution grant, so it is fetched on demand
and git-ignored rather than committed. **Do not commit third-party data without
a clear license** — follow the same download-and-verify pattern, or ask first.
