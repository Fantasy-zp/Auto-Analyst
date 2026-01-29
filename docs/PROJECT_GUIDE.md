# Auto-Analyst 项目指南

> 本文档面向只学过 Python 基础的读者，帮助你快速理解整个项目的结构和运行逻辑。

---

## 一、项目是做什么的？

Auto-Analyst（又名 DeepSeek-Insight Pro）是一个 **行业研究报告自动生成系统**。

用户输入一个研究主题（如"2025年中国低空经济"），系统会：
1. 自动搜索互联网上的相关资料
2. 用 AI 筛选出最相关的内容
3. 让 AI 基于这些资料撰写一份专业的行业研究报告

整个过程全自动，用户只需要等待几分钟就能得到一份包含数据和分析的研报。

---

## 二、项目文件结构

```
Auto-Analyst/
├── app.py              # 前端界面（用户看到的网页）
├── agent_manager.py    # 多智能体管理（研究员 + 撰稿人）
├── core_utils.py       # 核心工具（调用大模型和搜索引擎）
├── rag_processor.py    # RAG 处理（向量检索 + 重排序）
├── config.py           # 配置管理（API Key、参数等）
├── exceptions.py       # 自定义异常类
├── requirements.txt    # Python 依赖包列表
├── .env                # 环境变量（API Key，不提交到 Git）
├── .gitignore          # Git 忽略文件配置
├── README.md           # 项目说明文档
├── pytest.ini          # 测试配置
├── tests/              # 单元测试目录
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_core_utils.py
│   ├── test_rag_processor.py
│   └── test_exceptions.py
└── docs/               # 文档目录
    ├── PROJECT_GUIDE.md    # 本文件
    └── TECH_GUIDE.md       # 技术概念指南
```

---

## 三、每个文件的职责

### `app.py` — 前端界面

这是程序的入口文件。用 Streamlit 框架搭建了一个网页界面：
- 左侧边栏：显示运行状态、模式选择、知识库文件上传、功能按钮
- 主区域：输入研究主题、显示生成的报告、下载报告

**关键概念：** Streamlit 每次用户交互都会重新执行整个文件，所以用 `session_state` 保存需要持久化的数据。

### `agent_manager.py` — 多智能体管理

定义了两个 AI 智能体（Agent）和它们的协作方式：
- **研究员 Agent**：负责搜索和收集行业数据，配备了搜索工具
- **撰稿人 Agent**：负责把研究员的数据整理成专业研报

两个 Agent 按顺序工作：研究员先搜集数据 → 撰稿人再写报告。

### `core_utils.py` — 核心工具

封装了两个外部 API 的调用：
- **DeepSeek**：大语言模型（类似 ChatGPT），用来生成文本
- **Tavily**：搜索引擎 API，用来搜索互联网信息

提供三个核心能力：普通对话、流式对话（边生成边显示）、网络搜索。

### `rag_processor.py` — RAG 处理器

实现了"两阶段检索"：
1. **粗筛**：把搜索结果存入向量数据库，用向量相似度快速找出 10 条候选
2. **精筛**：用重排序模型对 10 条候选重新打分，只保留最相关的 3 条

还提供 `add_raw_texts()` 方法，用于将用户上传的文件（按段落分块后）存入向量库。

### `config.py` — 配置管理

集中管理所有配置项（API Key、模型参数、文件路径等），避免在各文件中散落配置值。

### `exceptions.py` — 自定义异常

定义了项目专属的异常类（如 `DeepSeekAPIError`、`VectorStoreError`），方便精确捕获和处理不同类型的错误。

---

## 四、数据流（程序执行流程）

### 标准模式（多智能体协作）

```
用户输入主题
    ↓
app.py 创建 IndustryAnalystCrew
    ↓
CrewAI 启动顺序执行
    ↓
研究员 Agent 开始工作
    ├── 调用 AdvancedRAGSearchTool
    │   ├── Tavily 搜索网页 → 得到 5 条结果
    │   ├── 存入 ChromaDB 向量数据库
    │   └── 向量检索（10条）→ FlashRank 重排序 → 返回 Top 3
    └── 输出结构化调研草案
    ↓
撰稿人 Agent 接收调研草案
    ├── 基于调研数据撰写 6 章节报告
    └── 输出最终 Markdown 报告
    ↓
app.py 显示报告 + 提供下载
```

### 快速模式（流式输出）

```
用户输入主题
    ↓
app.py 调用 generate_report_streaming()
    ├── Tavily 搜索 → 5 条结果
    ├── ChromaDB 存储 + 向量检索 + 重排序 → Top 5 上下文
    └── DeepSeek 流式生成报告（逐字显示）
    ↓
app.py 实时显示 + 保存到 session_state
```

### 文件上传流程

```
用户在侧边栏上传 PDF/TXT/MD 文件
    ↓
解析文件内容
    ├── PDF：PyPDF2 逐页提取文本
    └── TXT/MD：UTF-8 解码
    ↓
按段落分块（以 \n\n 为分隔符，过滤过短段落）
    ↓
检查 session_state.uploaded_file_names，跳过已上传过的文件
    ↓
调用 rag.add_raw_texts() 存入 ChromaDB，记录文件名到已上传集合
    ↓
后续研报生成时，检索会同时覆盖上传文件和网络搜索结果

注意：清空向量数据库时会同步重置已上传记录，允许重新上传。
```

### 两种模式的区别

| 特性 | 标准模式 | 快速模式 |
|------|---------|---------|
| 执行方式 | CrewAI 多智能体协作 | 单次 API 调用 |
| Agent 数量 | 2 个（研究员 + 撰稿人） | 0 个（直接调用模型） |
| 搜索次数 | 研究员自主搜索多次 | 固定搜索 1 次 |
| 输出方式 | 等待完成后一次性显示 | 边生成边显示（打字机效果）|
| 报告质量 | 更高（多角度调研） | 较好（单次搜索） |
| 速度 | 较慢 | 较快 |

---

## 五、如何运行项目

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的DeepSeek密钥
TAVILY_API_KEY=你的Tavily密钥
```

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

### 4. 运行测试

```bash
pytest
```

---

## 六、关键设计决策

### 为什么用 DeepSeek 而不是 OpenAI？
DeepSeek 的 API 兼容 OpenAI 格式，但成本更低。通过设置 `base_url` 指向 DeepSeek 服务器，用 OpenAI 的 SDK 就能调用 DeepSeek。

### 为什么需要 RAG（检索增强生成）？
大模型的知识有截止日期，不知道最新的行业数据。RAG 先从网上搜索最新信息，再让大模型基于这些信息写报告，确保内容准确且时效性强。

### 为什么要两阶段检索（向量搜索 + 重排序）？
向量搜索速度快但不够精准；重排序精准但速度慢。先用向量搜索快速缩小范围（10 条），再用重排序精选（3 条），兼顾速度和精度。

### 为什么用 session_state？
Streamlit 每次用户交互都会重新执行整个脚本，普通变量会丢失。`session_state` 是 Streamlit 提供的持久化存储，数据不会因 rerun 而丢失。

### 为什么用 @st.cache_resource？
ChromaDB 和 FlashRank 的初始化比较耗时（要加载模型文件）。`cache_resource` 确保这些资源只创建一次，后续直接复用。

### 为什么自定义异常？
用 `except DeepSeekAPIError` 比 `except Exception` 更精确，能区分是 API 错误还是搜索错误，日志也更清晰。面试时体现工程素养。

---

## 七、面试常见问题

1. **"介绍一下这个项目"** → 参见第一节
2. **"为什么选择这些技术栈？"** → 参见第六节
3. **"数据流是怎样的？"** → 参见第四节
4. **"RAG 是什么？为什么需要重排序？"** → 参见 `docs/TECH_GUIDE.md`
5. **"多智能体怎么协作的？"** → 参见 `agent_manager.py` 的注释
6. **"遇到了哪些技术难点？"** → LiteLLM 模型路由、Streamlit rerun 机制、依赖冲突处理
7. **"如何保证报告质量？"** → 多角度搜索 + RAG 两阶段检索 + 精心设计的 Prompt
