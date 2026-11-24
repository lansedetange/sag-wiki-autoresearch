# SAG-Wiki Autoresearch

结合 **SAG-Wiki 三层知识架构**（`Raw → Wiki → Index`）与自动化研究循环的私域 RAG 引擎骨架。项目保持单一进化文件，并提供可离线复现的检索评测与可选的 LLM 问答评测。

## 核心特性

- **三层知识架构**：只读原始证据层 `raw/`、结构化 Markdown 词条层 `wiki/`、全局导航索引 `index.json`。
- **双层检索**：先在 Wiki/Index 导航层定位主题，再下钻 Raw 证据层融合排序。
- **时间冲突处理**：查询现行规定时降低已废止证据的权重，避免旧版条款压过最新版本。
- **句级强制溯源**：LLM 回答中的证据 ID 必须来自本次检索结果，关键句缺少引用时整答拒答。
- **零幻觉熔断**：无达标证据时直接拒答；离线模式只返回明确标注的原始证据摘录。
- **缓存失效**：Raw 内容指纹或切分参数变化时自动重建 Wiki 与 Index。
- **单文件进化接口**：自动研究 Agent 只修改 `sag_wiki_engine.py`。

## 项目结构

```text
.
├── sag_wiki_engine.py       # 唯一进化文件：切分、编译、检索、问答
├── prepare.py               # 固定黄金集与严格评测裁判
├── program.md               # 夜间自动研究指令
├── fixtures/
│   ├── golden.jsonl         # 9 条固定评测样例
│   └── raw/                 # 固定原始证据
├── tests/
│   └── test_sag_wiki_engine.py
├── requirements.txt
└── README.md
```

运行后生成的 `sag_workspace/`、`demo_sag_workspace/` 和 `results.tsv` 均被 Git 忽略。

## 快速开始

要求 Python 3.9 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 离线演示：完成 Raw → Wiki → Index → 双层检索
python sag_wiki_engine.py

# 回归测试
python -m unittest discover -s tests -v

# 固定黄金集离线评测
python prepare.py --mode retrieval
```

无 API Key 时不会发起网络请求，也不需要安装 `openai` 包即可运行离线链路。配置 Key 后会使用 OpenAI 兼容接口编译 Wiki 和生成回答。

## 使用自己的知识库

```bash
mkdir -p sag_workspace/raw
# 将 .md 或 .txt 文档放入 sag_workspace/raw/

python sag_wiki_engine.py \
  --workspace ./sag_workspace \
  --question "2026 年上海出差住宿费上限是多少？"
```

默认只展示导航与证据检索结果。追加 `--answer` 可生成回答：

```bash
python sag_wiki_engine.py \
  --workspace ./sag_workspace \
  --question "2026 年上海出差住宿费上限是多少？" \
  --answer
```

修改 Raw 文档后会自动重建缓存；也可以显式执行：

```bash
python sag_wiki_engine.py --workspace ./sag_workspace --rebuild
```

## 启用 LLM 编译与问答

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

`OPENAI_BASE_URL` 可替换为任意兼容 OpenAI Chat Completions 协议的服务。默认模型为 `gpt-4o-mini`，可在 `sag_wiki_engine.py` 的 `CONFIG` 中修改。

完整评测会实际调用编译模型与问答模型：

```bash
python prepare.py --mode full --output results.tsv
```

也可以评测指定候选引擎：

```bash
python prepare.py --mode retrieval --engine /path/to/candidate.py
```

## 评测口径

`fixtures/golden.jsonl` 覆盖现行/废止条款冲突、跨文档问题、报销材料、事实查询和无答案查询。评测脚本不会使用网络检索，也不会让候选引擎修改评测集。

- **Retrieval Hit**：期望证据是否出现在 Top-K 中。
- **MRR**：首个期望证据在检索结果中的名次倒数。
- **Citation Valid**：回答引用是否非空，且全部属于本次检索证据。
- **Answer Correct**：非拒答题包含全部必需事实且带引用；拒答题必须正确拒答。

离线得分：

```text
score = 0.7 × retrieval_hit + 0.3 × MRR
```

完整问答得分：

```text
score = 0.35 × retrieval_hit
      + 0.15 × MRR
      + 0.20 × citation_valid
      + 0.30 × answer_correct
```

可用 `--min-score` 设置最低验收线，例如：

```bash
python prepare.py --mode retrieval --min-score 1.0
```

当前固定离线基线为 **9/9 命中、MRR 1.0、总分 1.0**。该结果只说明固定回归集通过，不代表任意私域文档上的泛化质量。

## 自动化进化

将 `program.md` 交给编码 Agent 即可开始快循环实验。核心约束是：

1. 只修改 `sag_wiki_engine.py`。
2. 不修改黄金集、评测脚本或测试。
3. 每轮只验证一个假设。
4. 单元测试全绿且离线得分不低于基线才保留。
5. 不得为了样例硬编码答案或证据 ID。

## 实现说明与限制

- 当前检索使用轻量中英文词法覆盖率与词频余弦相似度；`bm25_weight` 是词法部分的权重槽位，尚未接入真实 Embedding 或 Cross-Encoder。
- Raw 按字符切分，适合作为最小骨架；生产场景应加入文档结构感知切分和更大的评测集。
- Wiki 的 LLM 编译会将全部 Raw 语料发送给模型，超大数据集需要分批编译或上下文预算。
- 引用校验是保守的句级规则；格式高度特殊的回答可能被拒答。
- 本项目不包含用户认证、权限隔离、审计存储或生产级服务接口。
