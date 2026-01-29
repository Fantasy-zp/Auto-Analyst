# agent_manager.py
"""
CrewAI 多智能体管理模块

【Agent（智能体）是什么？】
Agent = 一个有角色、目标、能力的 AI 实体。
类比：你组建了一个团队，每个人有自己的职位和任务。
- 研究员 Agent：负责搜索和收集数据（有搜索工具）
- 撰稿人 Agent：负责写报告（没有工具，只用大模型能力）

【CrewAI 是什么？】
CrewAI 是一个多智能体编排框架，帮你：
1. 定义多个 Agent（角色、目标、工具）
2. 定义 Task（任务描述、期望输出）
3. 按顺序或并行执行任务
4. 自动把上一个任务的输出传给下一个任务

【LiteLLM 是什么？】
CrewAI 内部使用 LiteLLM 来调用不同的大模型。
LiteLLM 是一个"统一接口"，用同一种方式调用 OpenAI、DeepSeek、Claude 等不同模型。
格式：provider/model_name，如 "deepseek/deepseek-chat"

【Prompt Engineering（提示词工程）】
给 Agent 写好 role/goal/backstory 就是 Prompt Engineering：
- role：Agent 的职位（如"高级研究员"）
- goal：Agent 的目标（如"收集市场数据"）
- backstory：Agent 的背景故事（如"10年经验的分析师"）
这三者共同决定了 Agent 的行为模式和输出质量。

【依赖注入】
AdvancedRAGSearchTool 通过构造函数接收 core 和 rag 实例，
而不是自己创建。这样外部可以传入共享实例，避免重复创建资源。
"""
from crewai import Agent, Task, Crew, Process, LLM  # CrewAI 核心组件
from crewai.tools import BaseTool                     # 工具基类

from config import Config, get_logger
from core_utils import AnalystCore
from rag_processor import AdvancedRAG
from exceptions import SearchError, RAGError

logger = get_logger(__name__)

# 程序启动时设置环境变量（CrewAI/LiteLLM 需要通过环境变量获取 API Key）
Config.setup_openai_env()


class AdvancedRAGSearchTool(BaseTool):
    """
    高级 RAG 搜索工具 - Agent 用来搜索信息的"武器"

    【什么是 Tool？】
    Tool 是 Agent 可以使用的工具。Agent 自己不能上网搜索，
    但我们给它一个"搜索工具"，它就可以在需要时调用这个工具。
    类比：研究员自己不能出门，但有一台可以搜索的电脑。

    【这个工具做了什么？】
    1. 用 Tavily 搜索网页信息
    2. 把搜索结果存入 ChromaDB 向量库
    3. 从向量库中检索 + 重排序，返回最相关的内容

    【BaseTool 是什么？】
    CrewAI 提供的工具基类。继承它并实现 _run 方法，
    CrewAI 就知道怎么调用你的工具。
    """

    # CrewAI 要求的字段：工具名称和描述
    # Agent 会根据 description 判断什么时候该使用这个工具
    name: str = "advanced_rag_search_tool"
    description: str = "高级行业搜索工具，结合 Rerank 技术获取精准资讯。"

    # 内部依赖（通过构造函数注入）
    _core: AnalystCore = None
    _rag: AdvancedRAG = None

    def __init__(self, core: AnalystCore = None, rag: AdvancedRAG = None, **kwargs):
        """
        初始化工具，支持依赖注入

        【依赖注入是什么？】
        不让工具自己创建 AnalystCore 和 AdvancedRAG，
        而是从外部传入已有的实例。
        好处：多个组件可以共享同一个实例，避免重复创建。

        【object.__setattr__ 是什么？】
        CrewAI 的 BaseTool 基于 Pydantic，不能直接给"私有属性"赋值，
        所以用 object.__setattr__ 绕过 Pydantic 的限制。
        """
        super().__init__(**kwargs)
        object.__setattr__(self, '_core', core)
        object.__setattr__(self, '_rag', rag)

    def _run(self, query: str) -> str:
        """
        执行搜索 + RAG 处理（Agent 调用工具时会执行此方法）

        【执行流程】
        1. 用 Tavily 搜索 query 相关的网页
        2. 把搜索结果存入向量库
        3. 从向量库检索 + 重排序，返回最相关的内容

        Args:
            query: Agent 传入的搜索关键词

        Returns:
            经过重排序的精选上下文文本
        """
        try:
            # 延迟初始化：如果外部没有注入实例，就自己创建
            if self._core is None:
                object.__setattr__(self, '_core', AnalystCore())
            if self._rag is None:
                object.__setattr__(self, '_rag', AdvancedRAG())

            logger.info(f"RAG 工具执行搜索: {query[:50]}...")

            # 第一步：用 Tavily 搜索
            raw_results = self._core.search_industry_info(query)
            if not raw_results:
                logger.warning("搜索未返回结果")
                return "未找到结果。"

            # 第二步：把搜索结果存入向量库
            self._rag.add_documents(raw_results)

            # 第三步：从向量库检索 + 重排序
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
    """
    行业分析智能体团队 - 编排多个 Agent 协作完成研报

    【工作方式】
    1. 创建两个 Agent：研究员 + 撰稿人
    2. 创建两个 Task：调研任务 + 撰写任务
    3. 组成 Crew（团队），按顺序执行
    4. 研究员的输出会自动传给撰稿人作为输入

    【Process.sequential（顺序执行）】
    Task 1 完成 → 输出传给 Task 2 → Task 2 完成 → 最终结果
    （研究员搜集数据 → 撰稿人撰写报告）
    """

    def __init__(self, topic: str, core: AnalystCore = None, rag: AdvancedRAG = None):
        """
        初始化智能体团队

        Args:
            topic: 研究主题（如 "2025年中国低空经济"）
            core: AnalystCore 实例（可选，传入以复用）
            rag: AdvancedRAG 实例（可选，传入以复用）
        """
        self.topic = topic

        # 创建或复用资源实例
        # "or" 的作用：如果 core 不为 None 就用 core，否则创建新的
        self._core = core or AnalystCore()
        self._rag = rag or AdvancedRAG()

        # ---------- 配置 LLM（大语言模型）----------
        # 使用 CrewAI 原生的 LLM 类，它内部通过 LiteLLM 路由到 DeepSeek
        # model="deepseek/deepseek-chat" 告诉 LiteLLM：用 DeepSeek 服务商的 deepseek-chat 模型
        self._llm = LLM(
            model=Config.DEEPSEEK_MODEL,           # "deepseek/deepseek-chat"
            temperature=Config.DEEPSEEK_TEMPERATURE,  # 0.3，较保守的输出
            timeout=Config.DEEPSEEK_TIMEOUT           # 300 秒超时
        )

        # ---------- 创建搜索工具（注入依赖）----------
        self._rag_tool = AdvancedRAGSearchTool(core=self._core, rag=self._rag)

        # ---------- 创建研究员 Agent ----------
        # Prompt Engineering：通过 role/goal/backstory 精确控制 Agent 行为
        self._researcher = Agent(
            role='高级行业研究员',       # 角色定位
            goal=(                       # 具体目标（越详细，Agent 行为越可控）
                f'针对"{topic}"进行多维度深度调研，'
                '必须从市场规模、竞争格局、技术趋势、政策环境等多个角度分别搜索，'
                '收集具体的数据、事实和案例'
            ),
            backstory=(                  # 背景故事（帮助 AI 进入角色）
                '你是一名拥有10年经验的行业研究分析师，曾就职于顶级咨询公司。'
                '你的工作习惯是：针对一个课题，从至少3个不同角度进行搜索，'
                '确保信息覆盖全面。你特别擅长发现市场数据和关键趋势。'
                '你只输出经过验证的事实和数据，不编造信息。'
            ),
            tools=[self._rag_tool],      # 给研究员配备搜索工具
            verbose=True,                # 打印详细执行日志
            llm=self._llm,              # 使用 DeepSeek 模型
            allow_delegation=False       # 不允许委派任务给其他 Agent
        )

        # ---------- 创建撰稿人 Agent ----------
        self._writer = Agent(
            role='资深行业研报撰稿人',
            goal=(
                '基于研究员提供的调研数据，撰写一份结构清晰、'
                '论据充分、具备专业深度的行业研究报告'
            ),
            backstory=(
                '你是一名服务于投资机构的专业研报撰稿人，'
                '擅长将碎片化的行业数据整合为逻辑严密的深度报告。'
                '你的报告以数据驱动、结构清晰、结论明确著称。'
                '你会标注数据来源，确保内容可追溯。'
            ),
            verbose=True,
            llm=self._llm,
            allow_delegation=False
            # 注意：撰稿人没有 tools，只靠大模型能力 + 研究员的输出来撰写
        )

        logger.info(f"IndustryAnalystCrew 初始化完成，主题: {topic}")

    def run(self) -> str:
        """
        执行完整的分析流程

        【执行过程】
        1. 创建研究任务和撰写任务
        2. 组建 Crew 团队
        3. kickoff() 启动执行
        4. CrewAI 自动按顺序执行：研究 → 撰写
        5. 返回最终报告

        Returns:
            撰稿人生成的最终研报内容（Markdown 格式）
        """
        logger.info(f"开始执行行业分析任务: {self.topic}")

        # ---------- 创建研究任务 ----------
        # description：详细描述任务要求（这是影响输出质量最关键的 Prompt）
        # expected_output：告诉 Agent 期望输出的格式和内容
        research_task = Task(
            description=(
                f'请对"{self.topic}"进行全面深度调研，要求如下：\n'
                f'1. 使用搜索工具，从以下角度分别搜索（至少搜索3次）：\n'
                f'   - "{self.topic} 市场规模 数据"\n'
                f'   - "{self.topic} 竞争格局 头部企业"\n'
                f'   - "{self.topic} 最新政策 发展趋势"\n'
                f'2. 每次搜索后，提取关键数据和事实\n'
                f'3. 整理成结构化的调研草案，包含：\n'
                f'   - 核心数据（市场规模、增长率等具体数字）\n'
                f'   - 主要玩家及其市场份额\n'
                f'   - 关键技术或政策动态\n'
                f'   - 数据来源标注'
            ),
            expected_output=(
                '一份包含具体数据和事实的结构化调研草案，'
                '涵盖市场规模、竞争格局、技术趋势、政策环境等方面，'
                '每条信息标注数据来源'
            ),
            agent=self._researcher  # 分配给研究员执行
        )

        # ---------- 创建撰写任务 ----------
        # 撰稿人会自动收到研究员的输出作为上下文
        write_task = Task(
            description=(
                f'基于研究员提供的调研数据，撰写一份关于"{self.topic}"的专业行业研究报告。\n\n'
                '报告必须严格按照以下结构：\n'
                '## 一、行业概述\n'
                '简要介绍行业定义、范围和发展背景（200字以内）\n\n'
                '## 二、市场现状\n'
                '分析当前市场规模、增长率，引用具体数据\n\n'
                '## 三、竞争格局\n'
                '分析主要参与者、市场份额、竞争策略\n\n'
                '## 四、发展趋势\n'
                '识别3-5个关键趋势，每个趋势配以数据或案例支撑\n\n'
                '## 五、机遇与挑战\n'
                '分别列出主要机遇和挑战，给出具体分析\n\n'
                '## 六、投资建议\n'
                '基于以上分析给出明确的投资或发展建议\n\n'
                '要求：\n'
                '- 所有数据必须来自调研结果，不要编造\n'
                '- 使用 Markdown 格式\n'
                '- 报告总字数 1500-3000 字\n'
                '- 语言专业、客观，适合决策层阅读'
            ),
            expected_output=(
                '一份 1500-3000 字的 Markdown 格式行业研究报告，'
                '包含行业概述、市场现状、竞争格局、发展趋势、'
                '机遇与挑战、投资建议六个章节，数据准确、逻辑清晰'
            ),
            agent=self._writer  # 分配给撰稿人执行
        )

        # ---------- 组建并启动 Crew ----------
        crew = Crew(
            agents=[self._researcher, self._writer],     # 团队成员
            tasks=[research_task, write_task],            # 任务列表
            process=Process.sequential,                   # 顺序执行（研究 → 撰写）
            verbose=True                                  # 打印执行详情
        )

        # kickoff()：启动 Crew 执行所有任务
        # CrewAI 会：1. 让研究员执行研究任务 2. 把研究输出传给撰稿人 3. 撰稿人执行撰写任务
        result = crew.kickoff(inputs={'topic': self.topic})
        logger.info("行业分析任务执行完成")

        return str(result)
