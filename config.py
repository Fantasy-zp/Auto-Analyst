# config.py
"""
统一配置管理模块

【作用】
将所有配置项（API密钥、模型参数、路径等）集中在一个地方管理，
避免在各个文件中分散 hardcode（硬编码）配置值。
任何文件需要配置时，只需 from config import Config 即可。

【关键概念】
- dotenv：从 .env 文件加载敏感信息（如 API Key），避免密钥泄露到代码中
- logging：Python 内置日志模块，替代 print，支持日志级别和格式化
- 类变量：Config 类的属性在导入时就确定值，全局共享，无需实例化
"""
import os
import logging
from dotenv import load_dotenv

# ============================================================
# 第一步：加载 .env 文件中的环境变量
# ============================================================
# load_dotenv() 会读取项目根目录下的 .env 文件，
# 将其中的 KEY=VALUE 加载为环境变量（等效于在终端 export KEY=VALUE）
# 这样就可以通过 os.getenv("KEY") 获取值
load_dotenv()

# ============================================================
# 第二步：配置日志系统
# ============================================================
# logging.basicConfig 设置日志的全局格式：
# - level=INFO：只显示 INFO 及以上级别（DEBUG < INFO < WARNING < ERROR）
# - format：日志格式 = 时间 - 模块名 - 级别 - 消息内容
# 例如输出：2024-01-15 10:30:00 - core_utils - INFO - DeepSeek API 调用成功
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger（日志记录器）

    每个模块调用 get_logger(__name__) 获取自己的 logger，
    这样日志中会显示是哪个模块产生的日志，方便调试。

    Args:
        name: 通常传入 __name__（Python 自动填充当前模块名）

    Returns:
        该模块专属的 Logger 实例
    """
    return logging.getLogger(name)


class Config:
    """
    应用配置类 - 集中管理所有配置项

    【设计模式】
    使用类变量（而非实例变量），所有配置在 import 时就初始化完成，
    使用时直接 Config.DEEPSEEK_API_KEY 访问，无需创建实例。

    【为什么不用字典？】
    类变量有 IDE 自动补全和类型提示，字典没有。
    """

    # ========== API 密钥 ==========
    # 从环境变量读取，实际值在 .env 文件中配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")   # DeepSeek 大模型的 API 密钥
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")       # Tavily 搜索引擎的 API 密钥

    # ========== DeepSeek API 配置 ==========
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"     # DeepSeek API 的服务地址

    # 【重要】两个模型名的区别：
    # - DEEPSEEK_MODEL_NAME：用于直接通过 OpenAI SDK 调用（core_utils.py 中使用）
    # - DEEPSEEK_MODEL：用于 CrewAI 框架，需要加 "deepseek/" 前缀让 LiteLLM 路由到正确的服务商
    DEEPSEEK_MODEL_NAME = "deepseek-chat"              # OpenAI 兼容格式
    DEEPSEEK_MODEL = "deepseek/deepseek-chat"          # LiteLLM 路由格式

    DEEPSEEK_TEMPERATURE = 0.3   # 温度参数：0~1，越低输出越确定/保守，越高越随机/创造性
    DEEPSEEK_TIMEOUT = 300       # API 超时时间（秒），5分钟

    # ========== 存储路径 ==========
    CHROMA_DB_PATH = "./chroma_db"        # ChromaDB 向量数据库的本地存储目录
    FLASHRANK_CACHE_PATH = "./opt"        # FlashRank 重排序模型的缓存目录

    # ========== RAG 配置 ==========
    # RAG = Retrieval-Augmented Generation（检索增强生成）
    RAG_RETRIEVE_COUNT = 10               # 第一阶段：从向量库中召回多少条候选文档
    RAG_RERANK_TOP_K = 3                  # 第二阶段：重排序后只保留得分最高的 K 条
    RAG_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # FlashRank 使用的重排序模型名称
    RAG_COLLECTION_NAME = "industry_reports"        # ChromaDB 中的集合名称（类似数据库表名）

    # ========== Tavily 搜索配置 ==========
    TAVILY_SEARCH_DEPTH = "advanced"      # 搜索深度："basic" 或 "advanced"，advanced 会更深入
    TAVILY_MAX_RESULTS = 5                # 每次搜索返回的最大结果数

    # ========== 重试配置 ==========
    MAX_RETRIES = 3    # API 调用失败时最多重试次数（使用指数退避策略）

    # ========== CrewAI 遥测配置 ==========
    DISABLE_TELEMETRY = True   # 禁用 CrewAI 的遥测数据上报（保护隐私）

    @classmethod
    def setup_openai_env(cls):
        """
        设置 LiteLLM/CrewAI 需要的环境变量

        【为什么需要这个方法？】
        CrewAI 内部使用 LiteLLM 来路由不同的 LLM 服务商。
        LiteLLM 通过环境变量获取 API 密钥（如 DEEPSEEK_API_KEY）。
        这个方法在程序启动时调用一次，确保环境变量就绪。

        【classmethod 是什么？】
        类方法，不需要创建实例就能调用：Config.setup_openai_env()
        第一个参数 cls 代表类本身（类似 self 代表实例）
        """
        if cls.DEEPSEEK_API_KEY:
            # 设置 DeepSeek 的 API Key（LiteLLM 用这个变量名来识别）
            os.environ["DEEPSEEK_API_KEY"] = cls.DEEPSEEK_API_KEY
            # 同时设置 OpenAI 格式的环境变量（兼容其他组件）
            os.environ["OPENAI_API_KEY"] = cls.DEEPSEEK_API_KEY
            os.environ["OPENAI_API_BASE"] = cls.DEEPSEEK_BASE_URL
            os.environ["OPENAI_BASE_URL"] = cls.DEEPSEEK_BASE_URL

        if cls.DISABLE_TELEMETRY:
            # 禁用 CrewAI 的遥测数据收集
            os.environ["OTEL_SDK_DISABLED"] = "true"
            os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

    @classmethod
    def validate(cls) -> bool:
        """
        验证必需的 API Key 是否已配置

        在程序启动时调用，如果缺少关键配置则返回 False，
        前端会显示错误提示并阻止继续运行。

        Returns:
            True 表示配置完整，False 表示缺少必需的配置项
        """
        missing = []
        if not cls.DEEPSEEK_API_KEY:
            missing.append("DEEPSEEK_API_KEY")
        if not cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")

        if missing:
            logger = get_logger(__name__)
            logger.error(f"缺少必需的环境变量: {', '.join(missing)}")
            return False
        return True
