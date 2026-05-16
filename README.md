# casual_soft_sensor

面向因果约束与 DML 思路的工业软测量实验仓库。

## 快速开始

```bash
python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor.yaml
```

## 目录入口

- `src/`：模型与方法实现代码
- `scripts/`：训练、评估、因果发现与数据处理脚本
- `configs/`：实验与训练配置
- `docs/`：方法文档、工程规范、论文笔记
- `reports/`：实验总结与评审记录
- `results/`：本地运行输出（仓库仅保留 `.gitkeep`）
- `scratch/`：临时草稿与中间文件（不提交）
- `tests/`：测试代码与测试占位文件

## 文档导航

- 仓库结构与提交规则：`docs/engineering/repo_structure.md`
- 方法相关文档：`docs/method/`
- 研究与论文笔记：`docs/paper_notes/`
- 实验报告：`reports/experiments/`
- 评审与排查记录：`reports/reviews/`

## 说明

仓库已按 repo hygiene 进行结构整理，算法实现与训练流程位于 `src/`、`scripts/`、`configs/`，本次整理不改变其行为逻辑。
