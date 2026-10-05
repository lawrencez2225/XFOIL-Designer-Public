# Sources for the bundled runtime / 内置运行库源码

This directory contains the exact package build recipes, build scripts,
patches, and recipe licenses taken from the checksum-verified Conda packages.
`sources.json` identifies the corresponding upstream archives and SHA-256
checksums. Every listed archive was downloaded and checked on 2026-10-05.

此目录保存与内置运行库版本对应的包构建配方、构建脚本、补丁及其许可证。
上游源码地址与校验值在 `sources.json`，均已实际下载核对。运行程序时无需
读取这些文件；它们用于源码获取、许可证履行和审查。

## GCC / Fortran libraries

`libgcc_s.1.1.dylib`, `libgfortran.5.dylib`, and `libquadmath.0.dylib`
come from GCC 15.2.0. Full corresponding upstream source is available without
an account at:

- https://ftp.gnu.org/gnu/gcc/gcc-15.2.0/gcc-15.2.0.tar.gz
- https://mirrors.ocf.berkeley.edu/gnu/gcc/gcc-15.2.0/gcc-15.2.0.tar.gz

SHA-256: `7294d65cc1a0558cb815af0ca8c7763d86f7a31199794ede3f630c0d1b0a5723`.
The original package also names zlib 1.3.1 as a build input:
https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
(SHA-256 `9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23`).

`recipes/gcc/` contains the full parent source-build recipe, build/install
scripts, recipe licenses, and all referenced patches. The three packages
shared byte-identical parent recipes, so that material is stored once.
`recipes/gcc/outputs/{libgcc,libgfortran,libgfortran5}/` contains each rendered
output recipe and selected `conda_build_config.yaml`. The output `meta.yaml`
records the selected version and ordered patch list. Compiler/bootstrap and
platform SDK dependencies are documented in the parent recipe. Run the full
parent recipe with conda-build to rebuild the package outputs, not an output
recipe alone. Follow the rendered recipe's patch order for manual inspection.

GCC code and runtime libraries have different file-level license notices;
keep the notices in the source archive. The shipped runtime libraries are
GPL-3.0-only WITH GCC-exception-3.1. Both full texts accompany the binaries
under `../share/licenses/gcc-libs/` and `../share/licenses/libgfortran/`.
Removing debugging records and applying new ad-hoc signatures does not
change their source code. Checksums are in `../packaging.json`.

The GCC Runtime Library Exception does not remove the source-distribution
requirement for distribution of the library itself:
https://www.gnu.org/licenses/gcc-exception-3.1-faq.en.html
This snapshot supplies explicit, versioned source-download directions next
to the object code, with matching patches and scripts included locally,
using the network-source option in GPLv3 section 6(d):
https://www.gnu.org/licenses/gpl-3.0.html#section6
Maintainers must preserve equivalent access to this exact source for as
long as required; mirror the archive if its upstream locations disappear.
The large upstream archives are not required for daily calculations and
are not duplicated inside this checkout.

## X11 libraries and data

Recipes for libX11 1.8.13, libxcb 1.17.0, libXau 1.0.12, and libXdmcp 1.1.5
are in the matching `recipes/` directories. Source URLs and checksums are
in `sources.json`. Full upstream copyright and permission notices are under
`../share/licenses/{libX11,libxcb,libXau,libXdmcp}/COPYING`.
The libX11 COPYING covers the X11 locale, error, and color data as well.
The local relative Compose includes are recorded in `../../LICENSE-README.md`.

Conda package recipes carry their own BSD license notices; those files remain
with the recipes. This directory is third-party source material, not code
covered by the root Apache-2.0 license.

Generated package recipe comments naming an upstream CI checkout directory
were omitted. Unneeded package about/environment inventories were not copied.
Version, checksums, build settings, patches, scripts, and license notices were kept.
