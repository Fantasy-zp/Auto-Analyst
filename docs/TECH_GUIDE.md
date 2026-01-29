# Auto-Analyst 技术概念指南

> 本文档为只学过 Python 基础的读者解释项目中用到的所有技术概念。
> 每个概念都从"是什么 → 为什么需要 → 在项目中怎么用"三个角度讲解。

---

## 目录

1. [大语言模型（LLM）](#1-大语言模型llm)
2. [Agent（智能体）](#2-agent智能体)
3. [CrewAI（多智能体框架）](#3-crewai多智能体框架)
4. [LiteLLM（统一模型接口）](#4-litellm统一模型接口)
5. [Prompt Engineering（提示词工程）](#5-prompt-engineering提示词工程)
6. [RAG（检索增强生成）](#6-rag检索增强生成)
7. [向量与 Embedding（文本嵌入）](#7-向量与-embedding文本嵌入)
8. [ChromaDB（向量数据库）](#8-chromadb向量数据库)
9. [重排序（Reranking）](#9-重排序reranking)
10. [流式输出（Streaming）](#10-流式输出streaming)
11. [Streamlit（前端框架）](#11-streamlit前端框架)
12. [环境变量与 dotenv](#12-环境变量与-dotenv)
13. [依赖注入](#13-依赖注入)
14. [指数退避重试](#14-指数退避重试)
15. [异常分层设计](#15-异常分层设计)
16. [Python Generator（生成器）](#16-python-generator生成器)
17. [单元测试与 Mock](#17-单元测试与-mock)

---

## 1. 大语言模型（LLM）

### 是什么？
LLM = Large Language Model（大语言模型），如 ChatGPT、DeepSeek。
本质是一个经过海量文本训练的 AI 程序，能理解和生成自然语言文本。

### 为什么需要？
项目需要 AI 来撰写行业研究报告。大模型能理解"写一份关于低空经济的研报"这样的指令，并输出结构化的文章。

### 在项目中怎么用？
```python
# core_utils.py 中通过 OpenAI SDK 调用 DeepSeek
response = self.client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是专业分析师"},  # 设定角色
        {"role": "user", "content": "写一份行业研报"},     # 用户指令
    ]
)
answer = response.choices[0].message.content  # 获取 AI 回复
```

**类比：** 你给一个聪明的助手发微信消息，它回复你一段文字。`messages` 就是聊天记录，`response` 就是助手的回复。

---

## 2. Agent（智能体）

### 是什么？
Agent = 一个有角色、目标、工具的 AI 实体。
它不是简单地"你问我答"，而是能自主决策该做什么——比如"我需要先搜索，再整理数据"。

### 为什么需要？
写一份研报需要多个步骤（搜索资料、整理数据、撰写报告），一个 Agent 很难做好所有事。就像公司里，研究员负责调研，撰稿人负责写报告。

### 在项目中怎么用？
```python
# agent_manager.py 中定义 Agent
researcher = Agent(
    role='高级行业研究员',           # 角色（它是谁）
    goal='收集行业数据和事实',        # 目标（它要做什么）
    backstory='你有10年经验...',     # 背景故事（帮 AI 进入角色）
    tools=[search_tool],            # 工具（它能用什么）
    llm=deepseek_model              # 大脑（用哪个大模型思考）
)
```

**类比：** 你是老板，Agent 是你的员工。你给员工一个职位（role）、一个任务目标（goal）、一段工作经历（backstory），再配一台电脑（tools）。员工就知道该怎么干活了。

---

## 3. CrewAI（多智能体框架）

### 是什么？
CrewAI 是一个 Python 框架，帮你管理多个 Agent 之间的协作。
就像一个项目经理，负责安排谁先做、谁后做、怎么传递信息。

### 核心概念
| 概念 | 含义 | 类比 |
|------|------|------|
| Agent | 一个 AI 员工 | 团队成员 |
| Task | 一个具体任务 | 工作任务单 |
| Crew | 一个团队 | 项目组 |
| Process | 执行方式 | 工作流程 |

### 在项目中怎么用？
```python
# 创建团队并执行
crew = Crew(
    agents=[researcher, writer],              # 团队成员
    tasks=[research_task, write_task],         # 任务列表
    process=Process.sequential                 # 顺序执行（研究 → 撰写）
)
result = crew.kickoff()  # 启动！
```

**执行过程：**
1. CrewAI 把 `research_task` 交给 `researcher`
2. `researcher` 自主使用搜索工具完成调研
3. CrewAI 把研究员的输出传给 `writer`
4. `writer` 基于调研数据撰写报告
5. 返回最终报告

---

## 4. LiteLLM（统一模型接口）

### 是什么？
LiteLLM 是一个"翻译器"，让你用同一种方式调用不同公司的大模型（OpenAI、DeepSeek、Claude 等）。

### 为什么需要？
CrewAI 内部使用 LiteLLM 来调用大模型。不同公司的 API 格式不同，LiteLLM 统一了调用方式。

### 关键格式
```
模型名格式：provider/model_name

例如：
- "openai/gpt-4"         → 调用 OpenAI 的 GPT-4
- "deepseek/deepseek-chat" → 调用 DeepSeek 的模型
- "anthropic/claude-3"    → 调用 Anthropic 的 Claude
```

### 在项目中怎么用？
```python
# agent_manager.py 中
llm = LLM(model="deepseek/deepseek-chat")  # "deepseek/" 前缀告诉 LiteLLM 用哪家服务

# 注意与 core_utils.py 的区别：
# core_utils.py 直接用 OpenAI SDK → model="deepseek-chat"（不需要前缀）
# agent_manager.py 通过 CrewAI/LiteLLM → model="deepseek/deepseek-chat"（需要前缀）
```

---

## 5. Prompt Engineering（提示词工程）

### 是什么？
设计给 AI 的指令文本（Prompt），让 AI 按照你的期望输出结果。
Prompt 写得好不好，直接决定了 AI 输出的质量。

### 核心技巧

**1. 角色设定**
```
差的 Prompt：写一份报告
好的 Prompt：你是一位拥有10年经验的行业研究分析师，请撰写一份专业研报
```

**2. 具体指令**
```
差的 Prompt：分析这个行业
好的 Prompt：请从市场规模、竞争格局、技术趋势三个维度分析，每个维度引用具体数据
```

**3. 输出格式**
```
差的 Prompt：写报告
好的 Prompt：请按以下结构撰写（Markdown格式）：
## 一、行业概述（200字以内）
## 二、市场现状（引用具体数据）
...
```

### 在项目中怎么用？
Agent 的 `role`、`goal`、`backstory` 本质上就是 Prompt：
- `role` = "你是谁"
- `goal` = "你要做什么"
- `backstory` = "你有什么经验"

Task 的 `description` 和 `expected_output` 也是 Prompt：
- `description` = 详细的任务说明
- `expected_output` = 期望的输出格式

---

## 6. RAG（检索增强生成）

### 是什么？
RAG = Retrieval-Augmented Generation（检索增强生成）

核心思想：**先搜索，再回答**。
- 没有 RAG：AI 只能靠自己"记忆"中的知识回答（可能过时或编造）
- 有 RAG：AI 先从知识库中搜索真实信息，再基于搜索结果回答

### 为什么需要？
大模型的知识有截止日期（如训练数据只到 2024 年），不知道最新的行业数据。RAG 让 AI 能基于最新的网络搜索结果来写报告。

### 在项目中的完整流程
```
1. 搜索：Tavily 搜索 "2025年低空经济" → 得到 5 篇文章

2. 存储：把 5 篇文章存入 ChromaDB 向量数据库
   文章 → 转换为向量（一组数字）→ 存储

3. 检索：用户查询 → 转换为向量 → 在数据库中找最相似的 10 条

4. 重排序：用 FlashRank 对 10 条精细打分 → 只保留 Top 3

5. 生成：把 Top 3 文章作为"背景资料"发给大模型 → 生成报告
```

**类比：** 你写毕业论文时：
1. 先去知网搜论文（搜索）
2. 下载保存到文件夹（存储）
3. 翻看找出相关的（检索）
4. 精读选出最有用的 3 篇（重排序）
5. 参考这 3 篇写自己的论文（生成）

---

## 7. 向量与 Embedding（文本嵌入）

### 是什么？
**向量** = 一组数字，如 `[0.12, -0.34, 0.56, ..., 0.78]`（通常有几百个数字）

**Embedding（嵌入）** = 把文本转换为向量的过程

关键特性：**语义相似的文本，向量也相似**
- "人形机器人" → `[0.8, 0.3, -0.1, ...]`
- "仿人机器人" → `[0.79, 0.31, -0.09, ...]`（很接近！）
- "今天天气好" → `[-0.2, 0.7, 0.5, ...]`（很不同！）

### 为什么需要？
计算机不能直接比较"人形机器人"和"仿人机器人"是否相似，但可以计算两组数字的距离。把文本转为向量后，就能用数学方法找到语义相似的文档。

### 在项目中怎么用？
```python
# rag_processor.py 中
# ChromaDB 的 DefaultEmbeddingFunction 使用 all-MiniLM-L6-v2 模型
# 它能把任何文本转换为 384 维的向量
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# 存入文档时，ChromaDB 自动调用 embedding_fn 转换
collection.add(documents=["人形机器人市场..."])  # 自动转为向量并存储

# 查询时，也会自动转换查询文本
results = collection.query(query_texts=["机器人前景"])  # 自动找最相似的
```

---

## 8. ChromaDB（向量数据库）

### 是什么？
一个专门存储和搜索向量的数据库。普通数据库存文字，向量数据库存向量。

### 与普通数据库的区别
| | 普通数据库（如 MySQL） | 向量数据库（如 ChromaDB） |
|---|---|---|
| 存什么 | 文字、数字 | 向量（一组数字） |
| 怎么查 | 精确匹配（WHERE name='张三'） | 相似度搜索（找最像的） |
| 用途 | 存用户信息、订单等 | 存文档嵌入、图片特征等 |

### 在项目中怎么用？
```python
# 创建客户端（数据保存在本地 ./chroma_db 目录）
client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合（类似数据库中的"表"）
collection = client.get_or_create_collection(name="industry_reports")

# 存入文档（ChromaDB 自动转为向量）
collection.add(
    ids=["doc1", "doc2"],                    # 文档 ID
    documents=["文章内容1", "文章内容2"],       # 原文
    metadatas=[{"source": "url1"}, {"source": "url2"}]  # 元数据
)

# 搜索最相似的文档
results = collection.query(
    query_texts=["查询内容"],  # 查询文本（自动转为向量）
    n_results=10               # 返回 10 条最相似的
)
```

---

## 9. 重排序（Reranking）

### 是什么？
在向量搜索返回的候选结果中，用更精准的模型重新排序。

### 向量搜索 vs 重排序

| | 向量搜索 | 重排序 |
|---|---|---|
| 速度 | 快（毫秒级） | 慢（秒级） |
| 精度 | 中等 | 高 |
| 原理 | 分别计算查询和文档的向量，比较距离 | 同时看查询和文档，判断相关性 |
| 类比 | 只看书名判断是否相关 | 读完摘要再判断是否相关 |

### 为什么要两阶段？
假设有 10000 篇文档：
- 直接用重排序：太慢（要逐一精读 10000 篇）
- 只用向量搜索：不够精准
- 先向量搜索筛出 10 篇，再重排序 → 又快又准

### 在项目中怎么用？
```python
# rag_processor.py 中使用 FlashRank 重排序
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# 构建重排序请求
passages = [{"id": 0, "text": "文档1"}, {"id": 1, "text": "文档2"}, ...]
request = RerankRequest(query="人形机器人前景", passages=passages)

# 执行重排序（返回按相关性从高到低排序的结果）
results = ranker.rerank(request)
top3 = results[:3]  # 只取最相关的 3 条
```

---

## 10. 流式输出（Streaming）

### 是什么？
AI 生成文本时，不等写完再一次性返回，而是每写一个字就立刻发送。前端逐字显示，像打字机效果。

### 普通模式 vs 流式模式
```
普通模式：
用户等待... 等待... 等待... [3分钟后] 整篇报告一次性出现

流式模式：
"今" → "今天" → "今天的" → "今天的市" → ... [逐字出现]
```

### 在项目中怎么用？
```python
# core_utils.py 中
response = self.client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    stream=True  # 开启流式模式
)

# response 不再是完整回复，而是一个"流"
for chunk in response:          # 每次循环得到一小段
    text = chunk.choices[0].delta.content
    if text:
        yield text              # 用 yield 逐块返回

# app.py 中实时显示
placeholder = st.empty()
full_text = ""
for chunk in core.chat_with_deepseek_stream(prompt):
    full_text += chunk
    placeholder.markdown(full_text)  # 每收到一块就更新页面
```

---

## 11. Streamlit（前端框架）

### 是什么？
一个用纯 Python 构建网页应用的框架。不需要学 HTML/CSS/JavaScript，只写 Python 就能做出好看的网页。

### 核心特性

**1. 声明式 UI** — 写什么就显示什么
```python
st.title("标题")          # 显示大标题
st.text_input("输入框")    # 显示输入框
st.button("按钮")          # 显示按钮
st.markdown("**加粗**")   # 显示 Markdown 文本
```

**2. Rerun 机制** — 每次交互都重新执行
```python
# 用户每点一次按钮，整个 app.py 都会从头到尾重新执行
# 所以需要 session_state 保存数据
st.session_state.data = "这个不会丢"  # 跨 rerun 持久化
normal_var = "这个每次 rerun 都会重置"
```

**3. 缓存** — 避免重复计算
```python
@st.cache_resource   # 函数只执行一次，后续返回缓存
def create_database():
    return connect_to_db()  # 昂贵操作，只执行一次
```

**4. 布局**
```python
col1, col2 = st.columns([7, 3])  # 左右两栏，7:3 比例
with st.sidebar:                  # 侧边栏
    st.title("侧边栏内容")
```

---

## 12. 环境变量与 dotenv

### 是什么？
**环境变量**：操作系统级别的"全局变量"，任何程序都可以读取。常用来存储 API Key 等敏感信息。

**dotenv**：一个 Python 库，从 `.env` 文件加载环境变量。

### 为什么需要？
API Key 是敏感信息，不能直接写在代码中（否则上传 GitHub 后全世界都能看到）。放在 `.env` 文件中，再把 `.env` 加入 `.gitignore`，就不会泄露。

### 在项目中怎么用？
```
# .env 文件（不会被提交到 Git）
DEEPSEEK_API_KEY=sk-xxxx你的密钥xxxx
TAVILY_API_KEY=tvly-xxxx你的密钥xxxx
```

```python
# config.py 中
from dotenv import load_dotenv
load_dotenv()  # 读取 .env 文件，加载为环境变量

import os
api_key = os.getenv("DEEPSEEK_API_KEY")  # 从环境变量获取
```

---

## 13. 依赖注入

### 是什么？
不让一个组件自己创建它需要的依赖，而是从外部传入。

### 举例理解
```python
# ❌ 不好的做法：每个组件自己创建依赖
class SearchTool:
    def __init__(self):
        self.db = Database()      # 自己创建数据库连接
        self.searcher = Searcher() # 自己创建搜索器
        # 问题：每创建一个 SearchTool，就会多一个数据库连接

# ✅ 好的做法：从外部传入（依赖注入）
class SearchTool:
    def __init__(self, db, searcher):
        self.db = db              # 外部传入，大家共享一个
        self.searcher = searcher

# 使用时：
shared_db = Database()  # 只创建一次
tool1 = SearchTool(db=shared_db, searcher=s)  # 共享
tool2 = SearchTool(db=shared_db, searcher=s)  # 共享
```

### 在项目中怎么用？
```python
# agent_manager.py 中
# AnalystCore 和 AdvancedRAG 通过构造函数注入，多个组件共享同一个实例
crew = IndustryAnalystCrew(topic, core=shared_core, rag=shared_rag)
```

---

## 14. 指数退避重试

### 是什么？
API 调用失败时，不立刻重试，而是等待一段时间再试。等待时间指数增长：1秒 → 2秒 → 4秒。

### 为什么需要？
如果 API 暂时过载，立刻重试只会加重负担。等一会儿再试，服务器可能已经恢复。

### 在项目中怎么用？
```python
# core_utils.py 中
for i in range(3):  # 最多重试 3 次
    try:
        return call_api()  # 成功就直接返回
    except Exception:
        if i < 2:  # 不是最后一次
            sleep_time = 2 ** i  # 1秒, 2秒, 4秒
            time.sleep(sleep_time)
raise Error("重试 3 次仍然失败")
```

---

## 15. 异常分层设计

### 是什么？
定义一组有继承关系的异常类，让你能精确捕获不同类型的错误。

### 在项目中的异常层级
```
AutoAnalystError（所有错误的父类）
├── APIError → DeepSeekAPIError      # 大模型 API 错误
├── SearchError → TavilySearchError  # 搜索引擎错误
└── RAGError
    ├── VectorStoreError             # 向量数据库错误
    └── RerankError                  # 重排序错误
```

### 好处
```python
try:
    do_something()
except DeepSeekAPIError:
    print("大模型调用失败")     # 只捕获大模型错误
except SearchError:
    print("搜索失败")          # 捕获所有搜索相关错误
except AutoAnalystError:
    print("系统错误")          # 捕获所有业务错误（兜底）
```

---

## 16. Python Generator（生成器）

### 是什么？
用 `yield` 关键字的函数叫生成器函数。它不会一次性返回所有结果，而是每次 `yield` 一个值。

### 普通函数 vs 生成器
```python
# 普通函数：一次性返回所有结果（占用大量内存）
def get_all():
    results = []
    for i in range(1000000):
        results.append(i)
    return results  # 内存中有 100 万个数字

# 生成器：每次只产生一个值（省内存）
def get_one_by_one():
    for i in range(1000000):
        yield i  # 每次只产生一个，用完再产下一个
```

### 在项目中怎么用？
流式输出就是用生成器实现的：
```python
# core_utils.py
def chat_stream(self, prompt):
    for chunk in api_response:  # API 返回的流
        yield chunk.text        # 每收到一小段就 yield 出去

# app.py
for text in core.chat_stream(prompt):
    display(text)  # 逐段显示
```

---

## 17. 单元测试与 Mock

### 是什么？
**单元测试**：对代码中的每个"单元"（通常是函数）单独测试，确保它正确工作。

**Mock（模拟）**：在测试时，用假对象替代真实的外部服务（如 API），这样：
- 不需要真的调用 API（省钱、不依赖网络）
- 可以控制"API 返回什么"来测试各种场景

### 在项目中怎么用？
```python
# tests/test_core_utils.py 中
from unittest.mock import patch, MagicMock

@patch('core_utils.OpenAI')  # 用假的 OpenAI 替代真的
def test_chat(mock_openai):
    # 设置假的返回值
    mock_openai.return_value.chat.completions.create.return_value = \
        MagicMock(choices=[MagicMock(message=MagicMock(content="假回复"))])

    core = AnalystCore()
    result = core.chat_with_deepseek("测试")
    assert result == "假回复"  # 验证结果正确
```

### 运行测试
```bash
pytest                # 运行所有测试
pytest -v             # 显示详细信息
pytest --cov          # 显示代码覆盖率
```

---

## 总结：技术栈一览

| 技术 | 用途 | 对应文件 |
|------|------|---------|
| DeepSeek | 大语言模型，生成文本 | core_utils.py |
| Tavily | 搜索引擎 API | core_utils.py |
| CrewAI | 多智能体协作框架 | agent_manager.py |
| LiteLLM | 统一模型调用接口 | agent_manager.py |
| ChromaDB | 向量数据库 | rag_processor.py |
| FlashRank | 重排序模型 | rag_processor.py |
| PyPDF2 | PDF 文件解析 | app.py |
| Streamlit | 前端网页框架 | app.py |
| python-dotenv | 环境变量管理 | config.py |
| pytest | 单元测试框架 | tests/ |
