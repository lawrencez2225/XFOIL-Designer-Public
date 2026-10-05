# XFOIL-Designer

> This is a clean snapshot for publication, without private history or local data.
> Third-party licenses and corresponding-source access are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**English** · [简体中文](README.md)

A reproducible airfoil and wing analysis workbench built around the bundled
XFOIL 6.996 solver and MIT's AVL. It runs entirely on your own machine — no
server, no account, no upload.

> ### ⚠️ Platform requirement: Apple Silicon macOS
>
> The bundled `xfoil`, `pplot`, and `pxplot` binaries are **arm64 Mach-O
> executables** linked against macOS `.dylib` libraries. They do **not** run on
> Intel Macs, Linux, or Windows. See
> [Platform support](#platform-support) for what does and does not work, and
> for how to use your own build of XFOIL on another platform.

---

## What it does

- **Airfoil analysis** — Reynolds and Mach sweeps, solve directly at a target
  `CL`, boundary-layer inspection, convergence diagnostics.
- **Trustworthy results** — every run is checked for solver-domain evidence
  rather than trusting convergence flags alone, and the report says whether a
  result can be trusted. See [docs/TRUST_AND_SURROGATE.md](docs/TRUST_AND_SURROGATE.md).
- **Wing analysis** — AVL geometry modelling, planform and spanwise loads,
  profile-drag estimation.
- **Design search** — sample airfoil shapes, screen candidates with a surrogate
  model, then solve only the shortlist.
- **Local web workbench** — a Chinese-language browser UI bound to `127.0.0.1`
  only, protected by a per-session token.
- **Experiment comparison** — validate against measured data, with the
  provenance of each dataset recorded.

## Quick start (Apple Silicon macOS)

Prerequisites: [XQuartz](https://www.xquartz.org/), [Conda](https://docs.conda.io/),
and [Git LFS](https://git-lfs.com/).

```bash
git lfs install
git clone https://github.com/lawrencez2225/XFOIL-Designer-Public.git
cd XFOIL-Designer
git lfs pull
conda env create -f environment.yml
conda activate xfoil-work
python -m xfoil_mac --install-airfoils   # fetches the airfoil database
python xfoil_work.py
```

Then **double-click `启动工作台.command`** for daily use, or run
`python xfoil_work.py` for the text menu (menu **14** opens the workbench,
**15** shows history, **12** starts the wing wizard).

> **The airfoil database is not shipped.** `coord_seligFmt/` is the UIUC
> Airfoil Coordinates Database, which is copyrighted upstream with no
> redistribution grant, so the project downloads it on first use instead. The
> archive is verified against a pinned SHA-256 and unpacked into
> `coord_seligFmt/`. Check the state at any time with
> `python -m xfoil_mac --airfoil-status`; re-fetch with `--force`.

> **Git LFS matters.** The XFOIL runtime libraries are stored with Git LFS. If
> you clone without it — or download a ZIP from the web UI — those files arrive
> as ~130-byte text pointers instead of working libraries. Verify with:
>
> ```bash
> file XFOIL_6.996/runtime/lib/libX11.6.dylib
> ```
>
> A working library reports `Mach-O 64-bit dynamically linked shared library
> arm64`. If it says `ASCII text`, run `git lfs pull`.

### Installing as a Python package

From a clone, an **editable** install gives you the `xfoil-designer` console
command:

```bash
pip install -e .             # core: numpy, matplotlib, tqdm
pip install -e ".[surrogate]"  # adds scikit-learn, joblib for design search
```

Editable mode matters: the program locates `XFOIL_6.996/` by walking up from
the package directory, so the repository layout must stay intact. A regular
`pip install` into `site-packages` will not find the bundled solver.

### Installing AVL (optional, for wing analysis)

AVL is **not** bundled. Fetch the official MIT build with a pinned checksum:

```bash
python -m xfoil_mac --install-avl
```

Automatic installation is Apple-Silicon-only. On any other platform, pass your
own compatible build with `--avl-binary PATH`.

## Platform support

| Platform | Core airfoil analysis | Local workbench | Wing analysis (AVL) |
|---|---|---|---|
| Apple Silicon macOS | ✅ bundled binary | ✅ | ✅ `--install-avl` |
| Intel macOS | ❌ binary is arm64 | ⚠️ UI loads, solves fail | ⚠️ needs `--avl-binary` |
| Linux | ❌ binary is Mach-O | ⚠️ UI loads, solves fail | ⚠️ needs `--avl-binary` |
| Windows | ❌ | ❌ | ⚠️ needs `--avl-binary` |

Reusing the existing code on another platform requires supplying a working
XFOIL executable. The complete Fortran source ships in
`XFOIL_6.996/source/Xfoil/` for exactly this purpose; building it needs a
Fortran toolchain and X11, which this repository does not include.

## Where files live

| Path | Contents |
|---|---|
| `xfoil_runs/` | Your runs, presets, and imported files. Git-ignored. |
| `examples/` | Ready-to-use wing, design, and selection examples, plus sourced reference data |
| `coord_seligFmt/` | Airfoil coordinate database. Git-ignored and downloaded on demand — see below |
| `xfoil_mac/`, `tests/` | The program and its tests |
| `XFOIL_6.996/` | The solver and its runtime libraries. **Do not split this directory apart.** |

Git tracks source, examples, and documentation. Calculation results, personal
presets, caches, the airfoil database, and the local AVL install are never
pushed.

## Documentation

| Topic | Document |
|---|---|
| Day-to-day reference | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Full feature and parameter guide | [docs/USER_GUIDE.md](docs/USER_GUIDE.md) |
| Install, environment, Apple Silicon troubleshooting | [docs/MACOS_SETUP.md](docs/MACOS_SETUP.md) |
| Code structure | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Result trust and the surrogate model | [docs/TRUST_AND_SURROGATE.md](docs/TRUST_AND_SURROGATE.md) |

These documents are currently **Chinese only**. English translations are
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Verifying the installation

```bash
conda activate xfoil-work
MPLBACKEND=Agg XFOIL_NATIVE_TESTS=1 python -m unittest discover -s tests -v
```

Without `XFOIL_NATIVE_TESTS=1`, the tests that drive the real solver are
skipped; the remaining suite runs anywhere.

Experimental validation data comes from the
[Ladson data at the Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/naca0012_val.html).
Experimental error and software correctness are reported separately: **passing
the test suite does not mean every aerodynamic case has been validated.**

## License

The project's own code is **Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The bundled XFOIL solver is a separate work by Mark Drela and Harold Youngren
(MIT), distributed under the **GNU GPL**; its license text is at
`XFOIL_6.996/COPYING`. This project invokes XFOIL as an independent program
through a subprocess interface, so the two are aggregated rather than combined.

`coord_seligFmt/` is the **UIUC Airfoil Coordinates Database** (Michael Selig,
University of Illinois). The upstream site grants **no redistribution
license**, so the project does not bundle it — it is fetched on demand with
attribution recorded in `coord_seligFmt/install.json`. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full
component-by-component inventory and the reasoning.

## Contributing

Bug reports, platform ports, and English translations are all welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md).
