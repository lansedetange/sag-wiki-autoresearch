# SAG-Wiki Autoresearch Program

你是一个只允许修改 `sag_wiki_engine.py` 的编码研究 Agent。目标是在不降低安全性与可解释性的前提下，提高固定黄金集上的综合得分。

## 不可突破的边界

1. 不得修改 `prepare.py`、`fixtures/`、`tests/`、`program.md` 或评测指标。
2. 不得为了通过样例硬编码问题、答案、文档名或证据 ID。
3. 不得绕过引用校验、拒答保护或废止文档优先级。
4. 不得读取、打印或提交 API Key；不得访问与实验无关的外部服务。
5. 不得执行删除用户文件、强制推送、重置用户未提交改动等破坏性操作。
6. 每次只验证一个清晰假设；失败后仅回退本次实验自己的改动。

## 基线

先运行：

```bash
python -m unittest discover -s tests -v
python prepare.py --mode retrieval --output /tmp/sag-baseline.tsv
```

基线通过标准：31 项单元测试全部通过，离线检索得分不低于 1.0，且无评测错误。

## 快循环：无需调用 LLM

优先实验：

- 调整 `CONFIG` 中的检索权重、Top-K、相关阈值和时间优先级。
- 改进中英文分词、问句停用词和词法覆盖统计。
- 优化 Wiki 导航与 Raw 下钻的融合方式。
- 改进上下文重排，但不得改变证据内容。
- 提高废止条款、跨文档问题与无答案查询的区分能力。

每轮执行：

```bash
python -m unittest discover -s tests -v
python prepare.py --mode retrieval --output /tmp/sag-candidate.tsv --min-score 1.0
```

## 慢循环：Wiki 编译

只有快循环稳定后才实验：

- 调整 Raw 切分大小与重叠。
- 改进离线 Wiki 编译的词条组织。
- 改进 LLM 编译提示词、词条颗粒度和跨文档归并。
- 改进缓存失效条件，但不得掩盖 Raw 内容变化。

配置 `OPENAI_API_KEY` 后，可运行完整问答评测：

```bash
python prepare.py --mode full --output /tmp/sag-full.tsv
```

完整评测会产生模型 API 费用。模型端点可通过 `OPENAI_BASE_URL` 配置为兼容 OpenAI 协议的服务。

## 验收规则

- 单元测试必须全部通过。
- 候选离线得分必须不低于基线。
- 若某项指标上升但安全指标下降，视为失败。
- 完整问答模式下，引用有效率与答案正确率不得下降。
- 如果多个实验得分相同，优先选择改动更小、延迟更低、解释更清晰的方案。

## 实验记录

每次实验在回复中记录：

- 假设：本次改动预期改善什么。
- 改动：仅列出本次实验涉及的位置。
- 结果：测试数量、离线得分、检索命中率、MRR、错误数。
- 结论：保留、回退或继续观察。
