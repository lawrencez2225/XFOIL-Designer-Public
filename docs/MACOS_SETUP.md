# XFOIL-MAC：macOS Apple Silicon 配置与使用指南

代码模块职责见 [项目结构说明](ARCHITECTURE.md)。

3.0 的本地工作台、选型、性能地图、实验对照、离线交互界面和 AVL，见 [分析工作台使用指南](USER_GUIDE.md)。原有 XFOIL 环境依赖不变。

本文说明如何在 Apple Silicon Mac 上配置并运行本仓库。完成配置后，可以通过 Python 菜单或命令行执行单翼型、批量翼型计算，并使用 XQuartz/Xplot11 查看 XFOIL 原生几何、压力系数和压力矢量图。

计算流程、菜单、参数及结果说明统一见 [使用说明](USER_GUIDE.md)。本文仅保留安装与环境排障。

## 1. 适用范围

本指南仅适用于：

- Apple Silicon Mac：M1、M2、M3、M4 及后续 arm64 机型；
- 仓库内置的 `XFOIL_6.996/bin/xfoil`；
- 根目录 Python 入口 `xfoil_work.py`；
- macOS 自带的 zsh 或其他能够使用 Conda 的终端。

当前 XFOIL 是已签名的 arm64 Mach-O 二进制，不适用于 Intel Mac。项目不需要重新编译 XFOIL，也不需要使用 Rosetta。

> 日常推荐双击项目根目录的 `启动工作台.command`；`python xfoil_work.py` 保留文字菜单。旧的 `XFOIL_6.996/XFOIL_work.command` 已转发到同一套 Python 程序。

### 保持目录完整

程序会从 `xfoil_mac/` 所在位置向上查找 XFOIL，因此整个项目目录可以移动或改名，但不要拆散以下结构：

```text
XFOIL-Designer-Public/
├── xfoil_work.py
├── xfoil_mac/
│   ├── cli.py
│   ├── workflows.py
│   ├── execution.py
│   └── ...
├── environment.yml
├── XFOIL_6.996/
│   ├── bin/xfoil
│   └── runtime/lib/
├── coord_seligFmt/
└── tests/
```

只复制 `xfoil_work.py` 或只复制 XFOIL 二进制都无法正常运行；`XFOIL_6.996/runtime` 中包含它所需的 X11、Fortran 等动态库。该目录已去除编译工具，仅供运行；不要将它当成完整 Conda 环境。

## 2. 最短配置路径

已经安装 XQuartz、Conda 和 Git LFS 时，在终端执行：

```bash
git lfs install
git clone https://github.com/lawrencez2225/XFOIL-Designer-Public.git
cd XFOIL-Designer-Public
git lfs pull
conda env create -f environment.yml
conda activate xfoil-work
python -m xfoil_mac --install-airfoils
python xfoil_work.py
```

克隆公开仓库不需要账户权限。若已经通过其他方式取得完整项目目录，从 `cd XFOIL-Designer-Public` 开始执行即可。

`--install-airfoils` 下载翼型坐标数据库到 `coord_seligFmt/`。该数据库有版权且上游未授予分发许可，因此不随仓库分发，需要这一步才能使用数据库筛选等功能。状态可用 `python -m xfoil_mac --airfoil-status` 查看。

首次配置或遇到问题时，请继续阅读下面的完整步骤。

## 3. 安装并初始化 XQuartz

XFOIL 的原生图形窗口基于 X11，macOS 需要安装 XQuartz。

1. 打开 [XQuartz 官方网站](https://www.xquartz.org/)。
2. 从网站的 Quick Download 下载适用于当前 macOS 的最新版安装包。
3. 完成安装；若安装程序要求注销，注销并重新登录 macOS。
4. 在终端启动一次 XQuartz：

```bash
open -a XQuartz
```

确认应用位于项目所检查的标准路径：

```bash
test -d /Applications/Utilities/XQuartz.app && echo "XQuartz OK"
```

正常时输出 `XQuartz OK`。以后执行带图形的工作流时，`xfoil_work.py` 会自动打开 XQuartz，并在环境中缺少 `DISPLAY` 时自动使用 `:0`。

## 4. 安装 Conda

项目使用独立环境 `xfoil-work`，依赖 Python 3.11、NumPy、Matplotlib 和 tqdm。

如果电脑尚未安装 Conda，请参考 [Conda 官方 macOS 安装指南](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)，选择适用于 macOS Apple Silicon/arm64 的 Miniconda、Anaconda Distribution 或 Miniforge。不要安装 x86_64 版本。

安装完成后关闭并重新打开终端，然后检查：

```bash
conda --version
conda info
```

若 `conda` 已安装但 `conda activate` 不可用，执行：

```bash
conda init zsh
```

随后关闭并重新打开终端。

## 5. 获取完整仓库与 Git LFS 文件

### 5.1 安装 Git LFS

仓库中的 `XFOIL_6.996/runtime/**/*.dylib` 等文件由 Git LFS 管理。Git LFS 是独立于 Git 的程序；未安装时，动态库可能只被下载为很小的文本指针。

安装方式见 [GitHub 官方 Git LFS 指南](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=mac)。如果已安装 Homebrew，也可以执行：

```bash
brew install git-lfs
git lfs install
```

不使用 Homebrew 时，按 GitHub 指南下载 Git LFS，然后仍需执行：

```bash
git lfs install
```

### 5.2 克隆仓库

```bash
git clone https://github.com/lawrencez2225/XFOIL-Designer-Public.git
cd XFOIL-Designer-Public
git lfs pull
```

克隆公开仓库不需要账户权限；仓库若仍为私有，HTTPS 克隆需要拥有权限的 GitHub 账户。若没有权限，应由项目所有者提供完整目录；不要只下载单个 Python 文件。

浏览器下载的 ZIP 不一定包含 Git LFS 实体文件，因此优先使用 Git 克隆。取得目录后可以这样识别 LFS 是否完整：

```bash
file XFOIL_6.996/runtime/lib/libX11.6.dylib
```

正常结果应包含 `Mach-O 64-bit dynamically linked shared library arm64`。如果显示 `ASCII text` 或文件内容以 `version https://git-lfs.github.com/spec/v1` 开头，请安装 Git LFS 后在仓库根目录执行：

```bash
git lfs pull
```

## 6. 创建或更新 `xfoil-work` 环境

所有以下命令都应在仓库根目录，即包含 `environment.yml` 和 `xfoil_work.py` 的目录中执行。

### 第一次创建

```bash
conda env create -f environment.yml
conda activate xfoil-work
```

### 环境已经存在

```bash
conda env update -n xfoil-work -f environment.yml --prune
conda activate xfoil-work
```

检查 Python 架构和依赖：

```bash
python --version
python -c "import platform; print(platform.machine())"
python -c "import numpy, matplotlib, tqdm; print('Python dependencies OK')"
```

预期结果为 Python 3.11、`arm64` 和 `Python dependencies OK`。

环境的具体安装路径不需要固定。激活环境后使用 `python xfoil_work.py` 即可；程序不会硬编码 `/opt/anaconda3` 或某个用户名。

## 7. 验证 XFOIL 二进制和运行库

在仓库根目录执行：

```bash
file XFOIL_6.996/bin/xfoil
file XFOIL_6.996/runtime/lib/libX11.6.dylib
file XFOIL_6.996/runtime/lib/libgfortran.5.dylib
codesign -dv --verbose=2 XFOIL_6.996/bin/xfoil
```

应满足：

- `xfoil` 为 `Mach-O 64-bit executable arm64`；
- `libX11.6.dylib` 和 `libgfortran.5.dylib` 为 arm64 动态库；
- `codesign` 输出包含 `Signature=adhoc`；
- 二进制具有执行权限。

如果缺少执行权限：

```bash
chmod +x XFOIL_6.996/bin/xfoil
```

查看 XFOIL 声明的动态库依赖：

```bash
otool -L XFOIL_6.996/bin/xfoil
```

其中的 `@rpath/libX11.6.dylib`、`libgfortran.5.dylib` 和 `libquadmath.0.dylib` 由仓库内 `runtime/lib` 提供。通过 `xfoil_work.py` 启动时，脚本会自动设置进程所需的 `DYLD_LIBRARY_PATH`；通常不要把该变量永久写入 `~/.zshrc`。

最后检查 Python 命令行入口：

```bash
python xfoil_work.py --help
```

## 8. 常用环境变量

这些变量只需在确有需求时设置：

| 变量 | 作用 | 示例 |
|---|---|---|
| `XFOIL_LIVE_CPX_PAUSE` | 单次 Cp/压力矢量回放每帧停留秒数 | `0.8`、`2` |
| `XFOIL_BATCH_PREVIEW` | 设为 `0` 关闭批量原生 Xplot11 预览 | `0` |
| `XFOIL_DISABLE_BATCH_PREVIEW` | 设为 `1` 同样关闭批量预览 | `1` |
| `XFOIL_XQUARTZ_START_WAIT` | XQuartz 启动后等待秒数；较慢电脑可增大 | `4` |
| `XFOIL_CHORD_M` | Re/Mach 互算使用的弦长，单位 m | `0.5` |
| `XFOIL_AIR_DENSITY` | 互算使用的空气密度 | `1.225` |
| `XFOIL_AIR_VISCOSITY` | 互算使用的动力黏度 | `1.7894e-5` |
| `XFOIL_SOUND_SPEED` | 互算使用的声速，单位 m/s | `340.3` |
| `XFOIL_DEFAULT_RE` | Re 和 Mach 都未提供时的默认 Re | `1000000` |
| `XFOIL_DEFAULT_MACH` | Re 和 Mach 都未提供时的默认 Mach | `0.0` |

例如，修改互算弦长并减慢回放：

```bash
XFOIL_CHORD_M=0.5 XFOIL_LIVE_CPX_PAUSE=1.5 python xfoil_work.py
```

关闭批量图形预览但继续保存结果 PNG：

```bash
XFOIL_BATCH_PREVIEW=0 python xfoil_work.py --batch coord_seligFmt --re 1000000 --aseq -4 12 1 --force
```

## 9. 常见故障排查

### 9.1 `XQuartz is not installed`

程序固定检查：

```text
/Applications/Utilities/XQuartz.app
```

从 XQuartz 官网重新安装到标准位置，然后执行：

```bash
open -a XQuartz
```

### 9.2 XQuartz 已启动，但没有 Xplot11 窗口

依次尝试：

```bash
open -a XQuartz
export DISPLAY=:0
XFOIL_XQUARTZ_START_WAIT=4 python xfoil_work.py
```

检查是否曾设置 `XFOIL_BATCH_PREVIEW=0` 或 `XFOIL_DISABLE_BATCH_PREVIEW=1`。单次命令行计算要加入 `--keep-result-open` 才会执行完整原生回放。

### 9.3 `Library not loaded: @rpath/...`

不要直接双击或裸执行 `XFOIL_6.996/bin/xfoil`，应使用：

```bash
conda activate xfoil-work
python xfoil_work.py
```

然后检查 LFS 动态库：

```bash
git lfs pull
file XFOIL_6.996/runtime/lib/libX11.6.dylib
file XFOIL_6.996/runtime/lib/libgfortran.5.dylib
```

如果这些文件是 ASCII Git LFS 指针，说明 LFS 文件尚未下载。

### 9.4 `bad CPU type in executable` 或显示 `x86_64`

检查：

```bash
uname -m
python -c "import platform; print(platform.machine())"
file XFOIL_6.996/bin/xfoil
```

本指南要求结果均为 `arm64`。若终端在 Rosetta 下运行，请在 Finder 的终端应用“显示简介”中取消“使用 Rosetta 打开”，并安装 Apple Silicon 版 Conda。

### 9.5 macOS 阻止打开或文件带有 quarantine 属性

先确认仓库来源可信并检查签名：

```bash
codesign -dv --verbose=2 XFOIL_6.996/bin/xfoil
xattr -l XFOIL_6.996/bin/xfoil
```

如果项目来自可信来源、确实因浏览器下载而被隔离，可以只对项目内 XFOIL 目录移除 quarantine：

```bash
xattr -dr com.apple.quarantine XFOIL_6.996
```

不要对不明来源的二进制执行此操作。

### 9.6 `conda: command not found` 或依赖导入失败

重新打开终端；仍不可用时按 Conda 官方指南修复安装。Conda 可用后执行：

```bash
conda init zsh
conda env update -n xfoil-work -f environment.yml --prune
conda activate xfoil-work
python -c "import numpy, matplotlib, tqdm; print('OK')"
```

### 9.7 `Mach must be less than 1`

XFOIL 只接受亚声速自由来流 Mach。输入较小 Mach，或者只输入合理 Re，让程序根据默认空气参数估算 Mach。

### 9.8 计算不收敛

可以按以下顺序处理：

1. 将迭代上限提高，例如 `--iter 400`；
2. 缩小攻角步长，例如从 `--aseq -5 15 1` 改为 `--aseq -5 15 0.5`；
3. 从接近零攻角开始扫描；
4. 检查翼型坐标是否有重复点、乱序、交叉或异常尺度；
5. 查看该工况的 `polar.txt` 和算例目录中的 `attempts/<编号>/xfoil.log`。

接近失速的大攻角不收敛并不一定表示安装错误，也可能超出了 XFOIL 所采用附着流/弱分离模型的适用范围。

### 9.9 批量数据库运行过慢

先用少量翼型和较窄攻角范围验证配置，再运行完整 `coord_seligFmt`。不需要实时原生预览时设置：

```bash
XFOIL_BATCH_PREVIEW=0 python xfoil_work.py --batch coord_seligFmt --re 1000000 --aseq -4 12 1 --force
```

### 9.10 移动项目后找不到 XFOIL

确保 `xfoil_work.py`、`xfoil_mac/` 与 `XFOIL_6.996/` 仍位于同一项目根目录。项目整体可以移动，但不能只移动脚本或拆走 `runtime`。

## 10. 更新仓库和运行测试

更新前先确认没有未提交修改：

```bash
git status --short
```

工作区干净时拉取更新和 LFS 文件：

```bash
git pull --ff-only origin main
git lfs pull
conda env update -n xfoil-work -f environment.yml --prune
```

运行完整测试：

```bash
env PYTHONPYCACHEPREFIX=/tmp/xfoil_pycache \
  /opt/anaconda3/envs/xfoil-work/bin/python \
  -m unittest discover -s tests -v
```

上面的绝对路径是本项目维护环境的测试命令。其他电脑只要已激活 `xfoil-work`，也可以使用：

```bash
PYTHONPYCACHEPREFIX=/tmp/xfoil_pycache python -m unittest discover -s tests -v
```

测试通过后应输出 `OK`；测试数量会随功能与回归用例增加。
