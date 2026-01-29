# Auto-Analyst

基于多智能体协作与 Rerank 增强型 RAG 的行业深度分析系统。

## 功能特性

- **多智能体协作**：使用 CrewAI 框架，研究员 Agent 负责数据采集，撰稿人 Agent 负责报告生成
- **两阶段检索**：向量搜索（ChromaDB）+ 重排序（FlashRank），提升检索精度
- **实时网络搜索**：集成 Tavily AI 搜索引擎，获取最新行业资讯
- **流式输出**：支持实时流式生成报告，即时查看生成进度
- **双模式支持**：标准模式（多智能体协作）和快速模式（流式输出）
- **DeepSeek LLM**：使用 DeepSeek-V3 作为底层语言模型

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│                  (支持流式输出 & 双模式切换)                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│      标准模式             │    │      快速模式             │
│  (CrewAI 多智能体协作)    │    │   (直接流式 API 调用)     │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              ▼                               ▼
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
│  │  (支持流式输出)  │  │     │  │   FlashRank   │    │
│  └─────────────────┘  │     │  │  (Reranker)   │    │
└───────────────────────┘     │  └───────────────┘    │
                              └───────────────────────┘
```

## 项目结构

```
Auto-Analyst/
├── app.py              # Streamlit 前端应用（支持双模式）
├── agent_manager.py    # CrewAI 多智能体管理
├── core_utils.py       # DeepSeek 和 Tavily API 封装（含流式输出）
├── rag_processor.py    # RAG 处理（向量检索 + 重排序）
├── config.py           # 统一配置管理
├── exceptions.py       # 自定义异常
├── requirements.txt    # Python 依赖
├── pytest.ini          # 测试配置
├── tests/              # 单元测试
│   ├── conftest.py     # 测试 fixtures
│   ├── test_config.py
│   ├── test_core_utils.py
│   ├── test_rag_processor.py
│   └── test_exceptions.py
├── docs/               # 项目文档
│   ├── PROJECT_GUIDE.md   # 项目指南（面向初学者）
│   └── TECH_GUIDE.md      # 技术概念指南
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

### 4. 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并显示覆盖率
pytest --cov=. --cov-report=term-missing

# 运行特定测试文件
pytest tests/test_core_utils.py -v
```

## 生成模式

### 标准模式（多智能体协作）

- 使用 CrewAI 框架编排多个专业 Agent
- 研究员 Agent 负责深度数据采集
- 撰稿人 Agent 负责报告结构化撰写
- 效果更好，但耗时较长

### 快速模式（流式输出）

- 直接调用 DeepSeek API 流式生成
- 实时显示生成进度和内容
- 速度更快，适合快速预览

## 工作流程

### 标准模式流程

1. 用户输入研究主题
2. **研究员 Agent** 调用 `AdvancedRAGSearchTool`：
   - 使用 Tavily 搜索网络资讯
   - 将结果存入 ChromaDB 向量库
   - 通过 FlashRank 重排序返回精选内容
3. **撰稿人 Agent** 基于研究数据生成结构化研报
4. 前端展示报告，支持下载 Markdown 格式

### 快速模式流程

1. 用户输入研究主题
2. 系统实时搜索相关资料（Tavily）
3. RAG 检索和重排序（ChromaDB + FlashRank）
4. 流式生成报告，边生成边显示
5. 前端实时展示，支持下载

## 配置说明

所有配置集中在 `config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_MODEL` | `deepseek/deepseek-chat` | DeepSeek 模型（LiteLLM 格式） |
| `DEEPSEEK_MODEL_NAME` | `deepseek-chat` | DeepSeek 模型（OpenAI 格式） |
| `DEEPSEEK_TEMPERATURE` | `0.3` | 生成温度 |
| `RAG_RETRIEVE_COUNT` | `10` | 向量检索召回数量 |
| `RAG_RERANK_TOP_K` | `3` | 重排序后返回数量 |
| `TAVILY_MAX_RESULTS` | `5` | Tavily 搜索结果数 |
| `MAX_RETRIES` | `3` | API 调用重试次数 |

## 技术栈

- **前端**：Streamlit
- **Agent 框架**：CrewAI + LiteLLM
- **LLM**：DeepSeek-V3（支持流式输出）
- **搜索引擎**：Tavily AI
- **向量数据库**：ChromaDB
- **重排序模型**：FlashRank (ms-marco-MiniLM-L-12-v2)
- **测试框架**：pytest + pytest-cov

## 测试覆盖

项目包含完整的单元测试：

- `test_config.py` - 配置模块测试
- `test_core_utils.py` - API 调用测试（含流式输出）
- `test_rag_processor.py` - RAG 处理测试
- `test_exceptions.py` - 异常类测试

所有外部 API 调用均使用 Mock，可离线运行测试。

## 文档

- [项目指南](docs/PROJECT_GUIDE.md) - 面向 Python 初学者的项目结构和运行逻辑说明
- [技术概念指南](docs/TECH_GUIDE.md) - 项目涉及的 17 个技术概念详解（LLM、Agent、RAG、CrewAI 等）

## 许可证

MIT License
