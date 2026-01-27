# app.py
"""Streamlit 前端应用 - DeepSeek-Insight Pro"""
# 首先导入配置模块，确保环境变量最先加载
from config import Config, get_logger
from exceptions import AutoAnalystError

# 验证配置
if not Config.validate():
    import sys
    print("配置验证失败，请检查 .env 文件中的 API Key 设置")
    sys.exit(1)

import streamlit as st
from agent_manager import IndustryAnalystCrew
from rag_processor import AdvancedRAG

logger = get_logger(__name__)

# 页面配置
st.set_page_config(
    page_title="DeepSeek-Insight Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 美化
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
    .error-container {
        background-color: #fee;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #fcc;
    }
    </style>
    """, unsafe_allow_html=True)


# 初始化 session_state（用于跨 rerun 保持数据）
if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "report_topic" not in st.session_state:
    st.session_state.report_topic = None
if "rag_context" not in st.session_state:
    st.session_state.rag_context = None


# 初始化 RAG 处理器（使用缓存避免重复创建）
@st.cache_resource
def get_rag() -> AdvancedRAG:
    """获取缓存的 RAG 实例"""
    logger.info("创建 AdvancedRAG 实例")
    return AdvancedRAG()


# 侧边栏配置
with st.sidebar:
    st.title("运行状态")
    st.success("API 连接：DeepSeek-V3")
    st.success("搜索引擎：Tavily AI")
    st.divider()

    if st.button("清空向量数据库"):
        try:
            rag = get_rag()
            rag.clear_db()
            st.toast("向量数据库已清空")
            logger.info("用户清空了向量数据库")
        except AutoAnalystError as e:
            st.error(f"清空失败: {e}")
            logger.error(f"清空向量数据库失败: {e}")

    if st.button("清除当前报告"):
        st.session_state.report_result = None
        st.session_state.report_topic = None
        st.session_state.rag_context = None
        st.toast("报告已清除")
        logger.info("用户清除了当前报告")
        st.rerun()

# 主界面
st.title("DeepSeek-Insight Pro")
st.caption("基于多智能体协作与 Rerank 增强型 RAG 的行业深度分析系统")

topic = st.text_input(
    "调研课题",
    placeholder="例如：2025年中国低空经济发展趋势分析"
)

if st.button("开始生成深度研报"):
    if not topic:
        st.error("请输入调研课题名称")
    else:
        logger.info(f"用户发起研报生成请求: {topic}")

        try:
            # 获取 RAG 实例（复用缓存）
            rag = get_rag()

            with st.status("智能体团队正在执行多步任务...", expanded=True) as status:
                st.write("正在调用研究员 Agent 进行全网数据采集...")

                # 创建 Crew 并执行（传入 rag 实例以复用）
                crew_manager = IndustryAnalystCrew(topic, rag=rag)
                result = crew_manager.run()

                st.write("撰稿人 Agent 正在完成最后的报告润色...")
                status.update(
                    label="调研任务已全部完成！",
                    state="complete",
                    expanded=False
                )

            # 保存结果到 session_state
            st.session_state.report_result = result
            st.session_state.report_topic = topic

            # 同时保存 RAG 检索上下文
            try:
                st.session_state.rag_context = rag.retrieve_and_rerank(topic, top_k=3)
            except AutoAnalystError as e:
                st.session_state.rag_context = None
                logger.warning(f"获取 RAG 上下文失败: {e}")

            logger.info(f"研报生成完成: {topic}")

        except AutoAnalystError as e:
            error_msg = f"任务执行失败: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            st.markdown(
                '<div class="error-container">请检查 API 配置和网络连接，或稍后重试。</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            error_msg = f"发生未知错误: {e}"
            logger.error(error_msg)
            st.error(error_msg)


# 显示报告结果（从 session_state 读取，保证跨 rerun 持久化）
if st.session_state.report_result:
    st.divider()
    col_main, col_side = st.columns([7, 3])

    with col_main:
        st.markdown("### 最终行业研究报告")
        st.markdown(
            f'<div class="report-container">{st.session_state.report_result}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            label="下载 Markdown 格式报告",
            data=str(st.session_state.report_result),
            file_name=f"{st.session_state.report_topic}_DeepSeek_Insight.md",
            mime="text/markdown"
        )

    with col_side:
        st.markdown("### RAG 检索回溯")
        with st.expander("查看 Rerank 后的核心参考资料", expanded=True):
            if st.session_state.rag_context:
                st.info(st.session_state.rag_context)
            else:
                st.warning("暂无检索回溯数据")


st.markdown("---")
st.caption("面试项目演示 | Powered by DeepSeek-V3 & CrewAI")
