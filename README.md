# MarketOracle

MarketOracle 是一个在本机运行的 A 股单股研究助手。选择股票代码和研究基准日后，可以围绕行情、新闻、风险和公告连续追问，也可以让多空双方围绕证据展开辩论，再生成完整研报。它是研究辅助工具，不提供收益保证或交易指令。

## 先跑一次单股问答

以下命令从仓库根目录执行。后端需要能导入本项目所用 `hello_agents` 接口（包括 `HelloAgentsLLM` 和 `hello_agents.core.budget`）的 Python 环境，以及 `python-dotenv`、`pandas` 等依赖；真实行情与资讯工具还用到 `akshare`。当前仓库没有独立的后端依赖锁文件，如果已有运行本项目的 HelloAgents 环境，请优先使用它。

在本地创建 `.env`，按自己的模型服务填写，切勿提交真实密钥：

```dotenv
MARKETORACLE_MODEL=deepseek-chat
MARKETORACLE_MODEL_KEY=<你的模型密钥>
MARKETORACLE_MODEL_URL=<你的模型服务基础地址>
```

```sh
python -m backend.cli --symbol 000858.SZ --as-of 2025-12-31 --mock-data
```

进入后可输入「这只股票有哪些风险？」继续追问，或输入 `/研报 风险与分歧` 生成研报。`/专业` 和 `/小白` 切换回答风格，`exit` 退出。`--mock-data` 模拟的是行情、资讯等资料，**不是离线模型**；去掉它才会使用真实数据源。

## 使用 Web 界面

1. 在仓库根目录启动后端：`python -m backend.web`。健康检查 `GET http://127.0.0.1:8765/api/health` 正常时返回 `204`，没有响应正文。
2. 在另一个终端进入 `frontend`，依次运行 `npm ci` 和 `npm run dev`。
3. 打开 Vite 打印的本地地址。前端通过 `/api` 代理请求后端；只启动前端不能完成研究问答。

## 公告数据与检索

`backend/data/公告索引.csv` 是公告清单；`backend/data/rag/` 包含提取的页面、证据片段、处理报告及五粮液专题的 Chroma 向量索引。随仓库提供的是已处理的数据，**不含原始 PDF**。查阅片段不用重新下载 PDF；要核对原文或重新提取页面，则需要准备相应的源文件。向量检索另需兼容的 `chromadb`、`DASHSCOPE_API_KEY` 和 `MARKETORACLE_EMBEDDING_URL`，查询使用的嵌入模型应与索引匹配。

| 目录 | 用途 |
| --- | --- |
| `backend/agents/` | 单股对话、研究团队、多空辩论和研报裁决 |
| `backend/tools/` | 行情、新闻、风险提示与公告检索工具 |
| `backend/rag/` | 公告解析、证据校验、入库与查询 |
| `backend/tests/` | 后端问答、研究流程、Web 和公告检索测试 |
| `frontend/` | React、TypeScript 和 Vite 构建的本地研究界面 |

更多后端设计细节见[后端组件与协作说明](docs/后端组件与协作说明.md)。
