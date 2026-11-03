# CLAUDE.md

动手前请完整阅读：

1. **[docs/plan-for-agents.md](docs/plan-for-agents.md)**：项目计划、已核实的事实、研究问题、预注册草案、代码约定、里程碑和进度表。这是单一事实来源。
2. **[docs/guide-for-me.md](docs/guide-for-me.md)**：用户那边的操作步骤。你给出的指令要和它保持一致。

最重要的几条（完整版见 plan 第 1 节）：

- 不编造数字或引用；论文里每个数字都要能追溯到 `results/` 下的文件。
- `prereg/criteria.md` 在 RQ1 正式运行前提交；之后只能追加带日期的 Amendment。
- 不直接推 `main`；不提交模型权重或大文件。
- 花钱的操作（租卡、大规模 API 调用）先估算，再问用户。
- 和用户用中文交流；论文用英文写，最终由用户改写。
