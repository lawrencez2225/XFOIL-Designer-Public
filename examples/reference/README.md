# 实测基准来源

`ladson_naca0012.dat` 是 Ladson / NASA TM 4074 (1988) 的公开升阻力数据原文件，未修改。来源：https://tmbwg.github.io/turbmodels/NACA0012_validation/CLCD_Ladson_expdata.dat

背景说明：https://tmbwg.github.io/turbmodels/naca0012_val.html

条件：Re=6,000,000、Mach=0.15、有粗糙带强制转捩；80/120/180 grit 分成三组。不能把这三个系列混合。数据来源表没有给出转捩位置；程序的 Xtr 是单独记录的建模假设。原始 NACA0012 与 TMR 为 CFD 网格修改过的尖后缘几何并不相同。程序不模拟砂粒粗糙度阻力。

运行基准会复制原始文件，并在 `reference_provenance.json` 记录 SHA-256、所选系列、比较范围、几何 / 转捩假设。不要将其他求解器输出或程序自身结果称为实验数据。
