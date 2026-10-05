# XFOIL 运行依赖

此目录只供内置 Apple Silicon 求解器运行，不是完整 Conda 环境或编译环境。

- `lib/`：XFOIL、PPLOT、PXPLOT 所需的七个非系统动态库及其传递依赖：`libX11.6`、`libxcb.1`、`libXau.6`、`libXdmcp.6`、`libgfortran.5`、`libquadmath.0`、`libgcc_s.1.1`。
- `share/X11/`：保留 X11 的区域设置、颜色和错误信息数据。
- `share/licenses/`：保留 GCC / Fortran 完整 GPL、许可证例外及四个 X11 库的原始版权许可。
- `conda-meta/`：仅保留包标识、版本、来源和校验值；不是完整的 Conda 环境。
- `source/`：运行库对应源码的下载地址、校验值，以及原始包构建脚本、补丁和许可证；见 `source/README.md`。

动态库使用 Git LFS 保存。启动入口设置 `DYLD_LIBRARY_PATH`；系统动态库仍由 macOS 提供。不要单独移动 `lib/`。

旧编译器、链接工具、头文件、静态库和编译产物已删除。算法核对仍可使用上一级 `source/` 的原始 Fortran 源码；重新编译需要另行准备工具链。

## 二进制路径清理记录

三个求解器和 GCC / Fortran 库只移除了调试符号并重新签名，
所有架构的计算代码和数据区未改变。
libX11 使用经过校验的同版本原始发行包，通过 Conda 的标准二进制重定位
将构建前缀改为 XQuartz 的 `/opt/X11`；机器指令未改变。启动时，程序根据
当前位置设置 `XLOCALEDIR` 和 `XCMSDB`，优先读取随附的区域设置与颜色数据。
X11 错误信息数据库采用 XQuartz 的标准路径，外部自行安装的 XFOIL 不受影响。

`packaging.json` 记录原始包、源码及清理前后文件的校验值，用于核对分发文件。
此公开副本仅包含已清理的当前文件，没有复制原仓库的旧提交、旧 LFS 对象或私有备份。

验证包括完整测试、NACA 2412 在 TYPE 1/2/3 下的 15 个工况，以及移动到含空格
和中文目录后的 XQuartz、XFOIL、PPLOT、PXPLOT 绘图。气动力、Cp 和边界层
文件完全一致，三种图像的像素也完全一致。校验记录见 `packaging.json`。

PXPLOT 的验证采用其支持的 160 面板输入和已有的合法 dump 头；点数据均由
真实 XFOIL 计算。原版新建空二进制 polar dump 的 EOF 写入问题，以及 PXPLOT
每侧 132 点的数组上限仍然存在，本次没有修改 Fortran 算法或扩大数组。
