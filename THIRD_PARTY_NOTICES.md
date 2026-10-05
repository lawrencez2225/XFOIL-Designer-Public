# Third-party notices / 第三方组件声明

The root `LICENSE` and `NOTICE` cover the project's own Apache-2.0 code.
Third-party components retain their own copyright and license terms.
本项目自有代码采用 Apache-2.0；下列组件保留各自的版权与许可证。

## XFOIL and Xplot11

XFOIL 6.996 is by Mark Drela; its source headers explicitly state GPL-2.0
or later. Xplot11 is by Harold Youngren and Mark Drela; its source headers
state GNU Library GPL version 2 or later. This distribution uses the GPL-3.0
option for the combined executables. Full GPL-2.0, GPL-3.0, and GNU Library
GPL-2.0 texts are included in `XFOIL_6.996/` and its plotting source directory.

The Apple Silicon binaries are local builds, not official MIT macOS builds.
Complete source is in `XFOIL_6.996/source/Xfoil/`. The original archive from
https://web.mit.edu/drela/Public/web/xfoil/xfoil6.996.tgz is also included;
it was downloaded again and checked against the bundled archive.
The existing CSAV Cp export and plot-color changes, double-precision build
configuration, dated modification notices, and build instructions are in
`XFOIL_6.996/source/LOCAL_CHANGES.md`.

本目录包含本地构建的 macOS 程序及完整源码。XFOIL 的 GPL 版本来自源码
文件头，不是推测；绘图库有独立的 GNU Library GPL 声明。既有的 Cp 导出
与绘图修改已明确标注，不能把这些文件描述为未修改的官方二进制。

The Apache-2.0 Python application invokes the solver as a separate process
using command and result files. It does not relicense solver or library code.
Upstream: https://web.mit.edu/drela/Public/web/xfoil/

## Bundled runtime

| Component | Version | License | Full notices |
|---|---|---|---|
| libgcc, libgfortran, libquadmath | GCC 15.2.0 | GPL-3.0-only WITH GCC-exception-3.1 | `runtime/share/licenses/gcc-libs/`, `libgfortran/` |
| libX11 and X11 resource data | 1.8.13 | MIT-style, individual notices | `runtime/share/licenses/libX11/COPYING`, headers in resource files |
| libxcb | 1.17.0 | MIT | `runtime/share/licenses/libxcb/COPYING` |
| libXau | 1.0.12 | MIT | `runtime/share/licenses/libXau/COPYING` |
| libXdmcp | 1.1.5 | MIT | `runtime/share/licenses/libXdmcp/COPYING` |

Paths in the table are relative to `XFOIL_6.996/`.
Versioned upstream source URLs and checksums, exact package recipes,
build/install scripts, patches, and recipe licenses are under
`XFOIL_6.996/runtime/source/`. Source archives were downloaded and checksum
verified on 2026-10-05. Full GCC GPL and exception texts accompany the libraries.
The GCC runtime exception does not waive source access when distributing
the library itself; follow the source directions beside the binaries:
https://www.gnu.org/licenses/gcc-exception-3.1-faq.en.html

All ten native files have had embedded home-directory build paths removed.
Machine code is unchanged; libX11 resource strings use `/opt/X11` and launches
select bundled locale/color resources relative to the current directory.
Checksums and regression results are in `runtime/packaging.json`.
Fourteen Compose includes were changed to relative resource paths;
the original copyright notices were kept.

This clean export contains no old Git/LFS history, private backups, local
Conda environments, personal presets, or calculation outputs. License texts
and necessary source material must stay with future distributions.

## AVL and UIUC database: downloaded, not redistributed

AVL is obtained from https://web.mit.edu/drela/Public/web/avl/ using
`python -m xfoil_mac --install-avl`. Its license is GNU GPL. The ignored
`avl_runtime/` directory and downloaded binaries are not part of this export.
The installer verifies the official package checksum and records provenance.

The UIUC coordinate database by Michael Selig and the UIUC Applied Aerodynamics
Group is obtained from https://m-selig.ae.illinois.edu/ads/coord_database.html
using `python -m xfoil_mac --install-airfoils`. It has not granted this project
an open redistribution license; the ignored `coord_seligFmt/` is not included.
The installer verifies its pinned checksum and records the source URL.
Do not add either locally downloaded directory to the public repository.

## Experimental reference data

`examples/reference/ladson_naca0012.dat` comes from Ladson, NASA TM 4074 (1988),
via the NASA Turbulence Modeling Resource:
https://tmbwg.github.io/turbmodels/NACA0012_validation/CLCD_Ladson_expdata.dat
Its provenance and test conditions are in `examples/reference/README.md`.
This is attributed US Government research data. The three grit series must
not be mixed when comparing with numerical results.

## Python dependencies

NumPy, Matplotlib, tqdm, scikit-learn, and joblib are installed from their
upstream packages, not copied into this export. Their licenses remain those
distributed by the packages. Optional surrogate dependencies are loaded lazily.

If a component is missing or incorrectly attributed, please report it to
the project's maintainer.
