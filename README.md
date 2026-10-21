# AgentRec

AgentRec 是一个可部署的“协同召回 + 语义召回 + DIN 排序 + LLM Agent”推荐系统工程模板。仓库只保存代码和配置；服务器上的数据、模型、Embedding 和 FAISS 索引通过配置接入，不需要复制进 Git。

运行环境要求 Python 3.10 或更高版本。

## 系统架构

```text
FastAPI -> Agent workflow -> RecommendationService
                              |-- Two-Tower collaborative retrieval -> FAISS
                              |-- text semantic retrieval ----------> FAISS
                              `-- DIN ranker -> hard constraints -> metadata
             |-> OpenAI-compatible LLM (intent parsing and explanation only)
```

LLM 不负责凭空生成商品，也不判断价格等硬约束。所有最终商品均来自真实候选集，价格和类别约束由代码执行。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.api.main --check
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

Windows 激活命令为 `.venv\\Scripts\\activate`。未配置数据和模型时，API 仍可启动，`GET /health` 会显示 `degraded` 及缺少的资源；接入服务器资源后会变为 `ok`。

## 服务器接入

1. 编辑 `configs/server.yaml`，或设置环境变量：

```bash
export AGENTREC_ENV=server
export AGENTREC_DATA_ROOT=./dataset
export AGENTREC_ARTIFACT_ROOT=/artifacts/agentrec
export AGENTREC_LLM_BASE_URL=http://127.0.0.1:8000/v1
export AGENTREC_LLM_MODEL=/data/Qwen/Qwen3.5-9B
export AGENTREC_LLM_API_KEY=EMPTY
export AGENTREC_EMBEDDING_MODEL=/data/BAAI/bge-m3
```

2. 数据字段与路径都在环境配置的 `data` 段中。标准处理后文件为：

```text
${data_root}/processed/users.parquet
${data_root}/processed/items.parquet
${data_root}/processed/interactions.parquet
${data_root}/processed/train.parquet
${data_root}/processed/valid.parquet
${data_root}/processed/test.parquet
```

3. 训练并构建索引：

```bash
bash scripts/preprocess.sh server
bash scripts/train_retriever.sh server
bash scripts/build_index.sh server
bash scripts/train_ranker.sh server
bash scripts/evaluate.sh server
```

### 当前 All_Beauty 数据与 Qwen 配置

`configs/server.yaml` 已按 Amazon Reviews 2023 的 `All_Beauty` 原始 JSONL 配置：

```text
./dataset/All_Beauty.jsonl
./dataset/meta_All_Beauty.jsonl
```

交互商品键使用 `parent_asin`（它与元数据关联），评分使用 `rating`，元数据类别优先使用层级更细的 `categories`，缺失时回退到 `main_category`。`description` 和 `features` 的数组会在预处理阶段转成适合语义编码的文本。

Qwen3.5-9B 用于 Agent 的意图解析和推荐解释。若模型尚未通过 OpenAI 兼容接口启动，可在已安装 vLLM 的服务器上执行：

```bash
bash scripts/run_llm.sh
```

默认模型目录是 `/data/Qwen/Qwen3.5-9B`，服务地址是 `http://127.0.0.1:8000/v1`。Qwen 聊天模型不作为商品语义向量模型。当前已将 `/data/BAAI/bge-m3` 配置为 SentenceTransformer 稠密向量模型，启用向量归一化，并将默认最大长度设为 512、批大小设为 16，以控制首次构建商品向量时的显存占用。可根据服务器显存提高 `embedding.batch_size` 或 `embedding.max_length`。

所有 Python 入口也接受 `--env local|server` 和 `--config /path/to/override.yaml`。自定义配置会在默认配置之上递归合并。

## 数据格式

原始交互支持 JSON/JSONL、CSV 和 Parquet。字段名可配置，默认字段是：

| 语义 | 默认列 | 说明 |
|---|---|---|
| 用户 | `user_id` | 字符串或整数 |
| 商品 | `item_id` | 字符串或整数 |
| 时间 | `timestamp` | Unix 秒/毫秒或可解析日期 |
| 标签 | `label` | 可选，默认 1 |

商品元数据默认使用 `item_id/title/category/price/description`。预处理会清洗数据、迭代执行 user/item k-core、生成连续整数 ID，并按每个用户的时间顺序划分，避免把未来行为泄漏到训练集。

## 配置重点

- `configs/local.yaml`：本地调试与测试，默认 CPU。
- `configs/server.yaml`：服务器数据、模型和 OpenAI 兼容接口占位符。
- `configs/retriever.yaml`：Two-Tower 与召回参数。
- `configs/ranker.yaml`：DIN 参数。
- `configs/agent.yaml`：Agent、候选数、最终 Top-K 和约束策略。

环境变量优先级高于 YAML。完整变量列表见 `.env.example`。

## API

```text
GET  /health
GET  /items/{item_id}
POST /recommend   {"user_id":"42", "query":"1000元以下的轻量耳机", "top_k":10}
POST /chat        {"user_id":"42", "message":"1000元以下的轻量耳机", "top_k":10}
```

交互式接口文档位于 `/docs`。`/recommend` 返回结构化商品；`/chat` 额外执行意图解析、硬约束与解释生成。

## 训练产物

```text
${artifact_root}/checkpoints/two_tower.pt
${artifact_root}/checkpoints/din.pt
${artifact_root}/embeddings/collaborative.npy
${artifact_root}/embeddings/semantic.npy
${artifact_root}/faiss/collaborative.faiss
${artifact_root}/faiss/semantic.faiss
${artifact_root}/faiss/item_ids.json
```

索引读取器在没有 `faiss` 时可用 NumPy 做小规模精确检索，便于开发和测试；生产环境建议安装与服务器 CUDA 匹配的 FAISS 版本。

## 测试

```bash
pytest -q
python -m compileall src tests
```

测试覆盖时间切分、数据集、Two-Tower、DIN、混合召回、硬约束、顺序 Agent 流程及 API 健康检查。Notebook 仅用于分析，正式逻辑都在 `src/`。
