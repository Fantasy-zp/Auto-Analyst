# Auto-Analyst

基于多智能体协作与 Rerank 增强型 RAG 的行业深度分析系统。

## 功能特性

- **多智能体协作**：使用 CrewAI 框架，研究员 Agent 负责数据采集，撰稿人 Agent 负责报告生成
- **两阶段检索**：向量搜索（ChromaDB）+ 重排序（FlashRank），提升检索精度
- **实时网络搜索**：集成 Tavily AI 搜索引擎，获取最新行业资讯
- **DeepSeek LLM**：使用 DeepSeek-V3 作为底层语言模型

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  IndustryAnalystCrew                         │
│  ┌─────────────────┐         ┌─────────────────┐            │
│  │  Researcher     │────────▶│     Writer      │            │
│  │     Agent       │         │     Agent       │            │
│  └────────┬────────┘         └─────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │AdvancedRAGSearch│                                        │
│  │      Tool       │                                        │
│  └────────┬────────┘                                        │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────┐     ┌───────────────────────┐
│     AnalystCore       │     │     AdvancedRAG       │
│  ┌─────────────────┐  │     │  ┌───────────────┐    │
│  │  Tavily Search  │  │     │  │   ChromaDB    │    │
│  └─────────────────┘  │     │  │ (Vector Store)│    │
│  ┌─────────────────┐  │     │  └───────────────┘    │
│  │  DeepSeek Chat  │  │     │  ┌───────────────┐    │
│  └─────────────────┘  │     │  │   FlashRank   │    │
└───────────────────────┘     │  │  (Reranker)   │    │
                              │  └───────────────┘    │
                              └───────────────────────┘
```

## 项目结构

```
Auto-Analyst/
├── app.py              # Streamlit 前端应用
├── agent_manager.py    # CrewAI 多智能体管理
├── core_utils.py       # DeepSeek 和 Tavily API 封装
├── rag_processor.py    # RAG 处理（向量检索 + 重排序）
├── config.py           # 统一配置管理
├── exceptions.py       # 自定义异常
├── requirements.txt    # Python 依赖
├── chroma_db/          # ChromaDB 持久化目录
└── opt/                # FlashRank 模型缓存
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. 运行应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

## 工作流程

1. 用户输入研究主题
2. **研究员 Agent** 调用 `AdvancedRAGSearchTool`：
   - 使用 Tavily 搜索网络资讯
   - 将结果存入 ChromaDB 向量库
   - 通过 FlashRank 重排序返回精选内容
3. **撰稿人 Agent** 基于研究数据生成结构化研报
4. 前端展示报告，支持下载 Markdown 格式

## 配置说明

所有配置集中在 `config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_MODEL` | `deepseek/deepseek-chat` | DeepSeek 模型（LiteLLM 格式） |
| `DEEPSEEK_TEMPERATURE` | `0.3` | 生成温度 |
| `RAG_RETRIEVE_COUNT` | `10` | 向量检索召回数量 |
| `RAG_RERANK_TOP_K` | `3` | 重排序后返回数量 |
| `TAVILY_MAX_RESULTS` | `5` | Tavily 搜索结果数 |

## 技术栈

- **前端**：Streamlit
- **Agent 框架**：CrewAI + LiteLLM
- **LLM**：DeepSeek-V3
- **搜索引擎**：Tavily AI
- **向量数据库**：ChromaDB
- **重排序模型**：FlashRank (ms-marco-MiniLM-L-12-v2)

## 许可证

MIT License
