# 代码结构

根目录保留启动入口、README、环境与格式配置及功能目录。当前使用说明在 `docs/USER_GUIDE.md`；已被实现取代的路线图和旧设计笔记不再保留。

| 位置 | 职责 |
|---|---|
| `xfoil_work.py` / `xfoil_mac/__main__.py` | 兼容脚本和模块入口 |
| `xfoil_mac/cli.py` | 原有菜单与命令行 |
| `xfoil_mac/ui/commands.py` | 新工作台命令、历史操作 |
| `xfoil_mac/ui/service.py` | 统一配置、运行预检、任务编排 |
| `xfoil_mac/ui/server.py` | 仅本机 HTTP、鉴权、顺序队列和取消 |
| `xfoil_mac/ui/wing.py` | 机翼向导、参考几何、可旋转外观预览 |
| `xfoil_mac/ui/assets/` | 工作台与机翼预览页面，无 CDN |
| `xfoil_mac/storage/catalog.py` | 历史索引、预设、报告打包、目录边界 |
| `xfoil_mac/workflows.py` / `execution.py` | 原子清单、XFOIL 命令、重试、续算 |
| `xfoil_mac/runtime.py` / `solver_settings.py` | 本机依赖、环境和求解设置 |
| `xfoil_mac/flow.py` / `adaptive.py` | TYPE 1/2/3 工况与自适应攻角 |
| `xfoil_mac/data.py` / `results.py` | 数据解析、哈希、缺失点、唯一插值 |
| `xfoil_mac/analysis/` | CL 直接求解、边界层、诊断、数值敏感性、实验基准、设计搜索、机翼型阻 |
| `xfoil_mac/avl.py` / `wing_geometry.py` / `wing_plotting.py` | AVL 安装、输入、解析和外观图 |
| `xfoil_mac/plotting.py` / `reporting.py` / `comparison.py` | 图表、摘要与同条件比较 |
| `xfoil_mac/geometry.py` / `selection.py` / `maps.py` / `experiments.py` / `dashboard.py` | 保留的几何、筛选、地图、实验和离线查看器 |
| `tests/` | 单元、集成、真实求解器检查 |

`analysis/` 按实际计算分工：`lift.py` 直接 CL 求解，`boundary_layer.py` 边界层，`diagnostics.py` 收敛诊断，`studies.py` 面板/双向/实验检查，`design.py` 形状搜索，`coupling.py` 条带型阻积分，`report.py` 报告输出。保持现有导入路径，不增加通用任务框架。

内置求解器的 `runtime/lib/` 只保留对 XFOIL、PPLOT、PXPLOT 和本机 AVL 核对过的七个非系统动态库。`runtime/share/X11/` 保留图形数据，`share/licenses/` 与 `conda-meta/` 保留已有许可证和相关包来源记录；它不是完整 Conda 环境。编译器、静态库、头文件、目标文件及重复二进制已移除。原始 Fortran 源码保留用于核对算法。

## 数据约定

新任务 `job.json` 保存用户设置，计算证据位于 `data/`。原有 `run.json` 结构继续兼容，CL 和 AVL 分别使用 `lift_run.json`、`wing_run.json`，避免把不同求解模式混作同一种表格。报告只派生于保存的数据，失败点不补绘虚构曲线。

每次求解保留命令与原始输出；配置、输入内容、求解器和计算实现参与续算指纹。CL 直接求解还校验成功原始文件的哈希。面板、Ncrit、强制转捩进入对比和性能地图的分组策略。双向检查是独立实验，不覆盖原扫描。

本地服务只绑定 `127.0.0.1`，使用随机会话令牌，写请求检查 Host、Origin 和令牌；坐标文件限于项目，结果服务限于结果目录并防止路径 / 符号链接逃逸。后台运行固定 Python 入口和结构化配置，不拼接任意 shell。取消按进程组中断求解并保留清单。

## 开发验证

```sh
MPLBACKEND=Agg python -m unittest discover -s tests -v
MPLBACKEND=Agg XFOIL_NATIVE_TESTS=1 python -m unittest discover -s tests -v
```

软件检查验证程序行为；实验基准验证与具体测量条件的差异，不能相互替代。临时开发结果使用系统临时目录，日常结果通过报告首页查看。

格式统一为 PEP 8 的 79 列，物理符号和外部输出字段 `CL/CD/Re` 保持现有含义。Black 配置在 `pyproject.toml`，Flake8 配置在 `setup.cfg`；仅忽略与 Black 的 PEP 8 切片空格处理冲突的 E203，不忽略长行或未使用导入。

```sh
python -m black --check xfoil_mac tests xfoil_work.py
python -m flake8 xfoil_mac tests xfoil_work.py
MPLBACKEND=Agg XFOIL_HTTP_TESTS=1 python -m unittest tests.test_workbench_http -v
```

修改计算路径时，先保存正常算例的数值基准，再比较 CL、CD、CM、转捩位置和 AVL 系数；损坏/缺失输出单独用故障注入测试，不允许通过填默认值让它们成功。
