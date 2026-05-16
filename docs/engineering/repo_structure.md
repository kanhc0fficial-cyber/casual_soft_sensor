# Repository Structure & Submission Rules

## 目录用途

- `configs/`：训练、建模、因果发现、DML 与数据处理所需配置。
- `src/`：核心算法与模型实现代码。
- `scripts/`：训练、推理、评估、因果发现、数据处理脚本入口。
- `docs/method/`：方法说明、数据格式定义、设计文档。
- `docs/engineering/`：工程规范、仓库维护规则、开发说明。
- `docs/paper_notes/`：论文阅读、研究思路与方法笔记。
- `reports/experiments/`：阶段性实验结果总结、对比结论、简要指标汇总。
- `reports/reviews/`：临时检查记录、问题排查、评审与执行状态说明。
- `results/`：运行产物输出目录（仅保留 `.gitkeep` 作为目录占位）。
- `scratch/`：本地临时文件、草稿与一次性中间产物。
- `tests/`：测试代码与测试相关资源。

## 什么文件可以提交

- `src/`、`scripts/`、`configs/` 中的源码与配置（经评审后的正式修改）。
- `docs/` 下可长期维护的方法文档、工程文档、研究笔记。
- `reports/` 下的实验结论摘要与评审记录（建议 Markdown 或小型摘要 CSV）。
- `tests/` 下的测试代码。
- `results/.gitkeep`、`tests/.gitkeep` 等目录占位文件。

## 什么文件不应该提交

- `results/` 下完整运行产物（如大体量 `csv`、`pt`、`pth`、`pkl`、日志等）。
- `scratch/` 下任何临时文件。
- 机器本地缓存、检查点、临时日志与备份文件。

## results/ 与 scratch/ 使用规则

1. 所有训练/评估/推理运行输出统一写入 `results/`。
2. `results/` 默认不纳入版本控制，仅保留 `results/.gitkeep`。
3. 需要长期保留的实验结论，应整理为摘要文档并提交到 `reports/experiments/`。
4. `scratch/` 仅用于本地临时调试，不应提交任何具体文件。

## 后续实验输出规范

- 后续实验产生的模型、预测、日志、指标明细等文件统一放到 `results/`。
- 若需入库保存，请提炼为可复现、可审阅的摘要材料，放入 `reports/experiments/`。
