# XFOIL-MAC 气动工作台

> 本目录是用于公开的干净快照，只包含当前源码、示例、文档及已清理的运行文件。
> 第三方许可证和对应源码获取说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

[English](README.en.md) · **简体中文**

> ⚠️ **平台要求：Apple Silicon macOS。** 内置的 `xfoil` 是 arm64 Mach-O 二进制，
> **不能**在 Intel Mac、Linux 或 Windows 上运行。其他平台的可行方案见
> [英文说明的平台支持一节](README.en.md#platform-support)。

**日常使用：双击 `启动工作台.command`。** 浏览器会打开中文本地工作台，计算在这台 Mac 上运行。用完后关闭启动它的终端即可。

也可以激活 `xfoil-work` 环境后运行 `python xfoil_work.py`，进入原来的命令行菜单；菜单 **14** 打开工作台，**15** 查看历史，**12** 使用机翼建模向导。

工作台包含翼型 / Re / Mach 扫描、按 CL 直接求解、边界层分析、收敛诊断、双向逼近、面板敏感性、实测基准、AVL 机翼建模、机翼型阻估算，以及多工况翼型设计搜索。原来的数据库筛选、性能地图、实测 CSV 比较和原生 XFOIL 仍保留在菜单中。

| 想做什么 | 从哪里进入 |
|---|---|
| **日常速查（最先看这个）** | [日常速查](docs/QUICKSTART.md) |
| 开始计算、修改环境、保存预设 | `启动工作台.command` → 新建计算 |
| 看图、查看失败点、继续计算、打包报告 | 工作台 → 计算历史 |
| 找计算文件 | `xfoil_runs/index.html`；新版任务每次只有一个 `report.html` 首页 |
| 学习各功能与参数 | [完整使用说明](docs/USER_GUIDE.md) |
| 安装、环境和 Apple Silicon 排障 | [Mac 环境说明](docs/MACOS_SETUP.md) |
| 阅读或维护代码 | [代码结构](docs/ARCHITECTURE.md) |
| 判断结果能不能信、训练代理模型 | [数据可信度与代理模型](docs/TRUST_AND_SURROGATE.md) |

## 文件放在哪里

- **`xfoil_runs/`**：你的计算、预设、导入文件。`projects/` 是工作台计算。原有正式扫描和机翼目录保留原路径；开发测试使用系统临时目录。
- **`examples/`**：可直接使用的机翼、设计和筛选示例，以及有来源说明的实验数据。
- **`coord_seligFmt/`**：翼型坐标数据库。**不随仓库分发**，用 `python -m xfoil_mac --install-airfoils` 下载（见下）。
- **`xfoil_mac/`、`tests/`**：程序和测试，日常使用无需修改。
- **`XFOIL_6.996/`、`avl_runtime/`**：求解器及运行依赖，保留它们才能计算。内置 XFOIL 已精简为所需动态库、X11 数据和来源说明，不包含编译工具链。
- **`pyproject.toml`、`setup.cfg`**：代码格式和静态检查配置，日常计算无需修改。

源码、示例和文档纳入 Git；计算结果、个人预设、缓存、翼型数据库和本机 AVL 安装目录不推送。已有科学数据不因本次整理被删除。

## 翼型坐标数据库

`coord_seligFmt/` 是 UIUC 翼型坐标数据库（Michael Selig，伊利诺伊大学）。上游网站**没有给出分发许可**，所以本项目不再打包它，改为首次使用时下载：

```sh
python -m xfoil_mac --install-airfoils    # 下载并校验后解压到 coord_seligFmt/
python -m xfoil_mac --airfoil-status      # 查看当前状态
python -m xfoil_mac --install-airfoils --force   # 重新下载
```

压缩包按固定 SHA-256 校验，不匹配就拒绝；解压只接受 `.dat` 文件并把路径压平，因此构造过的压缩包无法写到目标目录之外。来源信息记录在 `coord_seligFmt/install.json`。

上游会就地更新该数据库，所以现在下载到的文件数（1665）比仓库原先携带的快照（1647）多，另有 155 个文件坐标被修订。**这不影响已有结果的可复现性**——每次计算都会把自己实际使用的几何保存为该次运行的 `geometry.dat`。

## 验证

```sh
conda activate xfoil-work
MPLBACKEND=Agg XFOIL_NATIVE_TESTS=1 python -m unittest discover -s tests -v
```

真实实验基准来自 [Turbulence Modeling Resource 的 Ladson 数据](https://tmbwg.github.io/turbmodels/naca0012_val.html)。实验误差与软件测试分别报告：通过测试不等于已验证所有气动工况。

## 安装为 Python 包（可选）

在克隆目录内做**可编辑安装**，即可获得 `xfoil-designer` 命令：

```sh
pip install -e .                # 核心：numpy、matplotlib、tqdm
pip install -e ".[surrogate]"   # 追加 scikit-learn、joblib，用于设计搜索
```

必须是可编辑安装：程序从包目录向上查找 `XFOIL_6.996/`，仓库结构不能拆散；装进 `site-packages` 的普通安装找不到内置求解器。

## 许可证

本项目自有代码采用 **Apache-2.0**，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

内置的 XFOIL 求解器是 Mark Drela 与 Harold Youngren（MIT）的独立作品，按 **GNU GPL** 分发，许可证全文见 `XFOIL_6.996/COPYING`。本项目通过子进程接口把 XFOIL 作为独立程序调用，两者属于聚合分发。

`coord_seligFmt/` 来自 **UIUC 翼型坐标数据库**（Michael Selig，伊利诺伊大学）。上游未给出分发许可，因此项目**不再打包**它，改为按需下载，来源信息记录在 `coord_seligFmt/install.json`。完整清单与理由见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

参与贡献见 [CONTRIBUTING.md](CONTRIBUTING.md)。
