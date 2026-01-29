# app.py
"""
Streamlit 前端应用 - DeepSeek-Insight Pro

【Streamlit 是什么？】
Streamlit 是一个用 Python 快速构建数据应用的框架。
你只需要写 Python 代码，Streamlit 自动帮你生成网页界面。
例如：st.title("标题") 就会在网页上显示一个大标题。

【Streamlit 的运行机制】
重要！Streamlit 的运行方式与普通 Python 脚本不同：
- 每次用户交互（点按钮、输入文字），Streamlit 都会 **从头到尾重新执行整个 app.py**
- 这叫做 "rerun"（重新运行）
- 所以如果不做特殊处理，变量每次都会被重置
- 解决方案：st.session_state（会话状态）和 @st.cache_resource（缓存）

【本文件的职责】
1. 配置网页界面（标题、布局、侧边栏）
2. 接收用户输入的研究主题
3. 根据模式（标准/快速）调用后端生成报告
4. 显示报告结果，提供下载功能
"""

# ============================================================
# 导入模块
# ============================================================
# 首先导入配置模块，确保环境变量最先加载
# 这一步必须在 agent_manager 之前，因为 agent_manager 导入时就会用到环境变量
from config import Config, get_logger
from exceptions import AutoAnalystError

import streamlit as st  # Streamlit 框架

logger = get_logger(__name__)

# ============================================================
# 页面基础配置（必须是 Streamlit 的第一个命令）
# ============================================================
# set_page_config 设置浏览器标签页标题、页面布局等
# layout="wide" 让页面占满整个浏览器宽度
st.set_page_config(
    page_title="DeepSeek-Insight Pro",
    layout="wide",
    initial_sidebar_state="expanded"  # 侧边栏默认展开
)

# ============================================================
# 验证配置
# ============================================================
# 检查 .env 文件中的 API Key 是否已配置
# 如果缺少，显示错误并停止运行（st.stop() 会终止当前脚本执行）
# 注意：不能用 sys.exit()，因为那会导致 Streamlit 整个进程退出
if not Config.validate():
    st.error("配置验证失败：请在 .env 文件中设置 DEEPSEEK_API_KEY 和 TAVILY_API_KEY")
    st.stop()

# 配置验证通过后，才导入需要 API Key 的模块
# agent_manager 在导入时就会调用 Config.setup_openai_env() 设置环境变量
from agent_manager import IndustryAnalystCrew
from rag_processor import AdvancedRAG
from core_utils import AnalystCore

# ============================================================
# CSS 样式美化
# ============================================================
# 通过注入 CSS 来美化页面外观
# unsafe_allow_html=True 允许渲染 HTML（仅用于样式，不含用户输入，安全）
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    .report-container {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 初始化 session_state（会话状态）
# ============================================================
# 【session_state 是什么？】
# Streamlit 每次 rerun 都会重置所有变量，但 session_state 不会。
# 它是一个跨 rerun 持久化的字典，用于保存需要保留的数据。
#
# 【为什么需要它？】
# 用户点击"下载报告"按钮时，Streamlit 会 rerun 整个脚本。
# 如果报告内容存在普通变量中，rerun 后就丢失了。
# 存在 session_state 中，rerun 后数据还在。
if "report_result" not in st.session_state:
    st.session_state.report_result = None      # 生成的报告内容
if "report_topic" not in st.session_state:
    st.session_state.report_topic = None       # 当前研究主题
if "rag_context" not in st.session_state:
    st.session_state.rag_context = None        # RAG 检索的参考资料


# ============================================================
# 缓存的资源实例
# ============================================================
# 【@st.cache_resource 是什么？】
# 装饰器，让函数只执行一次，后续调用直接返回缓存结果。
# 适用于"创建一次、反复使用"的资源（如数据库连接、模型实例）。
#
# 【与 session_state 的区别】
# - session_state：保存用户交互产生的数据（如报告内容）
# - cache_resource：保存昂贵的资源实例（如数据库连接），所有用户共享

@st.cache_resource
def get_rag() -> AdvancedRAG:
    """获取缓存的 RAG 实例（ChromaDB + FlashRank，只创建一次）"""
    logger.info("创建 AdvancedRAG 实例")
    return AdvancedRAG()


@st.cache_resource
def get_analyst_core() -> AnalystCore:
    """获取缓存的 AnalystCore 实例（DeepSeek + Tavily 客户端，只创建一次）"""
    logger.info("创建 AnalystCore 实例")
    return AnalystCore()


# ============================================================
# 快速模式的流式生成函数
# ============================================================
def generate_report_streaming(topic: str, rag: AdvancedRAG, core: AnalystCore):
    """
    流式生成研报（快速模式专用）

    【什么是流式生成？】
    普通模式：等 AI 写完整篇报告，一次性显示（可能等几分钟看不到任何内容）
    流式模式：AI 每写一个字就立刻显示，像打字机一样逐字出现

    【什么是 Generator（生成器）？】
    这个函数用了 yield 关键字，所以它是一个"生成器函数"。
    调用它不会立刻执行，而是返回一个生成器对象。
    外部通过 for 循环逐个获取 yield 出来的值：
        for chunk in generate_report_streaming(topic, rag, core):
            print(chunk)  # 每次循环获取一小段文本

    【执行流程】
    1. 用 Tavily 搜索相关资料
    2. 用 RAG（向量检索 + 重排序）筛选最相关内容
    3. 把筛选后的内容作为"背景资料"发给 DeepSeek，流式生成报告

    Args:
        topic: 研究主题（如 "2025年中国低空经济"）
        rag: AdvancedRAG 实例（用于向量检索和重排序）
        core: AnalystCore 实例（用于搜索和调用大模型）

    Yields:
        每次返回一小段文本（进度提示或报告内容片段）
    """
    # 第一步：用 Tavily 搜索网页信息
    yield "**[1/3] 正在搜索相关资料...**\n\n"
    search_results = core.search_industry_info(topic)

    if not search_results:
        yield "未找到相关资料，尝试直接生成分析...\n\n"
        context = ""
    else:
        yield f"找到 {len(search_results)} 条相关资料\n\n"

        # 第二步：RAG 处理（存入向量库 → 检索 → 重排序）
        yield "**[2/3] 正在进行智能检索和重排序...**\n\n"
        rag.add_documents(search_results)           # 存入 ChromaDB
        context = rag.retrieve_and_rerank(topic, top_k=5)  # 检索 + 重排序，取 Top 5
        yield "检索完成，已筛选出最相关的内容\n\n"

    # 第三步：流式生成报告
    yield "**[3/3] 正在生成行业研究报告...**\n\n"
    yield "---\n\n"  # 分隔线，区分进度提示和正式报告

    # 构建 Prompt：把背景资料 + 报告结构要求 发给 DeepSeek
    report_prompt = f"""你是一位专业的行业研究分析师。请根据以下背景资料，撰写一份关于"{topic}"的深度研究报告。

背景资料：
{context if context else "（无背景资料，请基于你的知识进行分析）"}

请严格按照以下结构撰写报告（使用 Markdown 格式）：

## 一、行业概述
简要介绍行业定义、范围和发展背景（200字以内）

## 二、市场现状
分析当前市场规模、增长率，引用具体数据

## 三、竞争格局
分析主要参与者、市场份额、竞争策略

## 四、发展趋势
识别3-5个关键趋势，每个趋势配以数据或案例支撑

## 五、机遇与挑战
分别列出主要机遇和挑战，给出具体分析

## 六、投资建议
基于以上分析给出明确的投资或发展建议

要求：
- 所有数据必须来自背景资料，不要编造
- 报告总字数 1500-3000 字
- 语言专业、客观，适合决策层阅读"""

    # 流式输出报告内容
    # chat_with_deepseek_stream 是一个生成器，每次 yield 一小段文本
    # 这里用 yield from 也可以，但 for + yield 更清晰
    for chunk in core.chat_with_deepseek_stream(report_prompt):
        yield chunk


# ============================================================
# 侧边栏（左侧面板）
# ============================================================
# with st.sidebar: 表示以下内容都渲染在左侧边栏中
with st.sidebar:
    st.title("运行状态")
    st.success("API 连接：DeepSeek-V3")     # 绿色提示框
    st.success("搜索引擎：Tavily AI")
    st.divider()  # 水平分隔线

    # ---------- 模式选择 ----------
    # st.radio 创建单选按钮组
    # format_func：把内部值（"standard"/"quick"）转换为用户可读的显示文本
    st.subheader("生成模式")
    generation_mode = st.radio(
        "选择模式",
        options=["standard", "quick"],
        format_func=lambda x: "标准模式（多智能体协作）" if x == "standard" else "快速模式（流式输出）",
        index=0,  # 默认选第一个（标准模式）
        help="标准模式使用 CrewAI 多智能体协作，效果更好但耗时较长；快速模式直接流式生成，速度更快"
    )

    st.divider()

    # ---------- 功能按钮 ----------
    # 清空向量数据库：删除 ChromaDB 中存储的所有搜索结果
    if st.button("清空向量数据库"):
        try:
            rag = get_rag()
            rag.clear_db()
            st.toast("向量数据库已清空")  # toast：右下角弹出的短暂提示
            logger.info("用户清空了向量数据库")
        except AutoAnalystError as e:
            st.error(f"清空失败: {e}")
            logger.error(f"清空向量数据库失败: {e}")

    # 清除当前报告：清空 session_state 中保存的报告数据
    if st.button("清除当前报告"):
        st.session_state.report_result = None
        st.session_state.report_topic = None
        st.session_state.rag_context = None
        st.toast("报告已清除")
        logger.info("用户清除了当前报告")
        st.rerun()  # 手动触发 rerun，刷新页面

# ============================================================
# 主界面
# ============================================================
st.title("DeepSeek-Insight Pro")
st.caption("基于多智能体协作与 Rerank 增强型 RAG 的行业深度分析系统")

# 文本输入框：用户输入研究主题
topic = st.text_input(
    "调研课题",
    placeholder="例如：2025年中国低空经济发展趋势分析"
)

# ============================================================
# 生成报告按钮的逻辑
# ============================================================
# st.button 返回 True 仅在用户点击的那一次 rerun 中
# 所以生成的结果必须存入 session_state，否则下次 rerun 就丢了
if st.button("开始生成深度研报"):
    if not topic:
        st.error("请输入调研课题名称")
    else:
        logger.info(f"用户发起研报生成请求: {topic}，模式: {generation_mode}")

        try:
            # 获取缓存的实例（不会重复创建）
            rag = get_rag()
            core = get_analyst_core()

            if generation_mode == "quick":
                # ========== 快速模式：流式输出 ==========
                st.markdown("### 正在生成研究报告...")

                # st.empty() 创建一个占位符，后续可以用 .markdown() 更新内容
                # 每次调用 report_placeholder.markdown() 会替换（而非追加）内容
                report_placeholder = st.empty()
                full_report = ""

                # 遍历生成器，逐块追加文本并实时更新页面
                for chunk in generate_report_streaming(topic, rag, core):
                    full_report += chunk
                    report_placeholder.markdown(full_report)  # 实时更新显示

                # 提取报告正文（去掉进度提示部分）
                # "---" 是分隔线，报告正文在最后一个 "---" 之后
                report_content = full_report.split("---\n\n")[-1] if "---" in full_report else full_report

                # 保存到 session_state，确保跨 rerun 持久化
                st.session_state.report_result = report_content
                st.session_state.report_topic = topic

                # 保存 RAG 上下文（用于右侧"检索回溯"面板）
                try:
                    st.session_state.rag_context = rag.retrieve_and_rerank(topic, top_k=3)
                except AutoAnalystError:
                    st.session_state.rag_context = None

                st.success("报告生成完成！")
                logger.info(f"快速模式研报生成完成: {topic}")

            else:
                # ========== 标准模式：CrewAI 多智能体协作 ==========
                # st.status 创建一个可折叠的进度指示器
                with st.status("智能体团队正在执行多步任务...", expanded=True) as status:
                    st.write("正在调用研究员 Agent 进行全网数据采集...")

                    # 创建 Crew 团队并执行（传入缓存实例以复用资源）
                    # 内部流程：研究员搜索 → RAG 处理 → 撰稿人撰写报告
                    crew_manager = IndustryAnalystCrew(topic, core=core, rag=rag)
                    result = crew_manager.run()  # 执行完整流程，返回最终报告

                    st.write("撰稿人 Agent 正在完成最后的报告润色...")
                    # 更新进度指示器为"完成"状态
                    status.update(
                        label="调研任务已全部完成！",
                        state="complete",
                        expanded=False
                    )

                # 保存结果到 session_state
                st.session_state.report_result = result
                st.session_state.report_topic = topic

                # 保存 RAG 检索上下文
                try:
                    st.session_state.rag_context = rag.retrieve_and_rerank(topic, top_k=3)
                except AutoAnalystError as e:
                    st.session_state.rag_context = None
                    logger.warning(f"获取 RAG 上下文失败: {e}")

                logger.info(f"标准模式研报生成完成: {topic}")

        except AutoAnalystError as e:
            # 捕获所有业务异常（API 错误、搜索错误、RAG 错误等）
            error_msg = f"任务执行失败: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            st.info("请检查 API 配置和网络连接，或稍后重试。")
        except Exception as e:
            # 兜底：捕获所有未预料的异常
            error_msg = f"发生未知错误: {e}"
            logger.error(error_msg)
            st.error(error_msg)


# ============================================================
# 显示报告结果（始终从 session_state 读取）
# ============================================================
# 这段代码在按钮逻辑之外，每次 rerun 都会执行
# 只要 session_state 中有报告，就会显示（不会因 rerun 而消失）
if st.session_state.report_result:
    st.divider()
    # st.columns 创建水平布局，[7, 3] 表示左侧占 70%，右侧占 30%
    col_main, col_side = st.columns([7, 3])

    with col_main:
        st.markdown("### 最终行业研究报告")
        # st.container(border=True) 创建带边框的容器，比 unsafe_allow_html 更安全
        with st.container(border=True):
            st.markdown(st.session_state.report_result)

        # 下载按钮：点击后浏览器会下载一个 .md 文件
        st.download_button(
            label="下载 Markdown 格式报告",
            data=str(st.session_state.report_result),
            file_name=f"{st.session_state.report_topic}_DeepSeek_Insight.md",
            mime="text/markdown"
        )

    with col_side:
        st.markdown("### RAG 检索回溯")
        # st.expander 创建可折叠面板，展示 RAG 检索到的参考资料
        with st.expander("查看 Rerank 后的核心参考资料", expanded=True):
            if st.session_state.rag_context:
                st.info(st.session_state.rag_context)  # 蓝色信息框
            else:
                st.warning("暂无检索回溯数据")  # 黄色警告框


# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.caption("面试项目演示 | Powered by DeepSeek-V3 & CrewAI")
