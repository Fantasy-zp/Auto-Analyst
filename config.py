# config.py
"""统一配置管理模块"""
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger"""
    return logging.getLogger(name)


class Config:
    """应用配置类"""

    # API Keys
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # DeepSeek API
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    # 原始模型名（用于直接调用 OpenAI 兼容 API）
    DEEPSEEK_MODEL_NAME = "deepseek-chat"
    # LiteLLM 格式（用于 CrewAI）
    DEEPSEEK_MODEL = "deepseek/deepseek-chat"
    DEEPSEEK_TEMPERATURE = 0.3
    DEEPSEEK_TIMEOUT = 300

    # 存储路径
    CHROMA_DB_PATH = "./chroma_db"
    FLASHRANK_CACHE_PATH = "./opt"

    # RAG 配置
    RAG_RETRIEVE_COUNT = 10
    RAG_RERANK_TOP_K = 3
    RAG_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
    RAG_COLLECTION_NAME = "industry_reports"

    # Tavily 配置
    TAVILY_SEARCH_DEPTH = "advanced"
    TAVILY_MAX_RESULTS = 5

    # 重试配置
    MAX_RETRIES = 3

    # CrewAI 遥测配置
    DISABLE_TELEMETRY = True

    @classmethod
    def setup_openai_env(cls):
        """设置 LiteLLM/CrewAI 环境变量"""
        if cls.DEEPSEEK_API_KEY:
            # LiteLLM 使用 DEEPSEEK_API_KEY 环境变量
            os.environ["DEEPSEEK_API_KEY"] = cls.DEEPSEEK_API_KEY
            # 保留 OpenAI 兼容配置（用于其他组件）
            os.environ["OPENAI_API_KEY"] = cls.DEEPSEEK_API_KEY
            os.environ["OPENAI_API_BASE"] = cls.DEEPSEEK_BASE_URL
            os.environ["OPENAI_BASE_URL"] = cls.DEEPSEEK_BASE_URL

        if cls.DISABLE_TELEMETRY:
            os.environ["OTEL_SDK_DISABLED"] = "true"
            os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

    @classmethod
    def validate(cls) -> bool:
        """验证必需的配置项"""
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
