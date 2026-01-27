# agent_manager.py
"""CrewAI 多智能体管理模块"""
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from config import Config, get_logger
from core_utils import AnalystCore
from rag_processor import AdvancedRAG
from exceptions import SearchError, RAGError

logger = get_logger(__name__)

# 设置 OpenAI 兼容环境变量（供 CrewAI 内部使用）
Config.setup_openai_env()


class AdvancedRAGSearchTool(BaseTool):
    """高级 RAG 搜索工具，结合 Rerank 技术获取精准资讯"""

    name: str = "advanced_rag_search_tool"
    description: str = "高级行业搜索工具，结合 Rerank 技术获取精准资讯。"

    # 通过类属性注入依赖，避免每次调用时重复创建
    _core: AnalystCore = None
    _rag: AdvancedRAG = None

    def __init__(self, core: AnalystCore = None, rag: AdvancedRAG = None, **kwargs):
        super().__init__(**kwargs)
        # 允许外部注入实例，或延迟创建
        object.__setattr__(self, '_core', core)
        object.__setattr__(self, '_rag', rag)

    def _run(self, query: str) -> str:
        """
        执行搜索和 RAG 处理

        Args:
            query: 搜索查询词

        Returns:
            经过重排序的精选上下文
        """
        try:
            # 延迟初始化（如果未注入）
            if self._core is None:
                object.__setattr__(self, '_core', AnalystCore())
            if self._rag is None:
                object.__setattr__(self, '_rag', AdvancedRAG())

            logger.info(f"RAG 工具执行搜索: {query[:50]}...")

            raw_results = self._core.search_industry_info(query)
            if not raw_results:
                logger.warning("搜索未返回结果")
                return "未找到结果。"

            self._rag.add_documents(raw_results)
            context = self._rag.retrieve_and_rerank(query, top_k=Config.RAG_RERANK_TOP_K)

            logger.info("RAG 工具执行完成")
            return context

        except (SearchError, RAGError) as e:
            logger.error(f"RAG 工具执行失败: {e}")
            return f"搜索处理失败: {e}"
        except Exception as e:
            logger.error(f"RAG 工具发生未知错误: {e}")
            return f"处理失败: {e}"


class IndustryAnalystCrew:
    """行业分析智能体团队"""

    def __init__(self, topic: str, core: AnalystCore = None, rag: AdvancedRAG = None):
        """
        初始化智能体团队

        Args:
            topic: 研究主题
            core: AnalystCore 实例（可选，用于复用）
            rag: AdvancedRAG 实例（可选，用于复用）
        """
        self.topic = topic

        # 创建或复用资源实例
        self._core = core or AnalystCore()
        self._rag = rag or AdvancedRAG()

        # 配置 LLM（使用 CrewAI 原生 LLM 类，支持 LiteLLM 格式）
        self._llm = LLM(
            model=Config.DEEPSEEK_MODEL,
            temperature=Config.DEEPSEEK_TEMPERATURE,
            timeout=Config.DEEPSEEK_TIMEOUT
        )

        # 创建工具实例（注入依赖）
        self._rag_tool = AdvancedRAGSearchTool(core=self._core, rag=self._rag)

        # 创建 Agent（每个 Crew 实例独立）
        self._researcher = Agent(
            role='高级行业研究员',
            goal=f'针对关键词 {topic} 搜集深度事实数据',
            backstory="你是一名严谨的数据专家，擅长从噪音中提取黄金信息。",
            tools=[self._rag_tool],
            verbose=True,
            llm=self._llm,
            allow_delegation=False
        )

        self._writer = Agent(
            role='专业资深撰稿人',
            goal='将研究数据转化为一份专业的结构化行业研报',
            backstory="你擅长撰写给决策层看的研报，风格专业。",
            verbose=True,
            llm=self._llm,
            allow_delegation=False
        )

        logger.info(f"IndustryAnalystCrew 初始化完成，主题: {topic}")

    def run(self) -> str:
        """
        执行分析任务

        Returns:
            最终研报内容
        """
        logger.info(f"开始执行行业分析任务: {self.topic}")

        # 创建任务（每次 run 创建新任务）
        research_task = Task(
            description=f"对 {self.topic} 进行深度调研。",
            expected_output="调研草案",
            agent=self._researcher
        )

        write_task = Task(
            description="撰写研报。",
            expected_output="最终研报",
            agent=self._writer
        )

        # 创建并运行 Crew
        crew = Crew(
            agents=[self._researcher, self._writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff(inputs={'topic': self.topic})
        logger.info("行业分析任务执行完成")

        return str(result)
