# exceptions.py
"""自定义异常模块"""


class AutoAnalystError(Exception):
    """Auto-Analyst 基础异常类"""
    pass


class ConfigError(AutoAnalystError):
    """配置错误"""
    pass


class APIError(AutoAnalystError):
    """API 调用错误"""
    pass


class DeepSeekAPIError(APIError):
    """DeepSeek API 调用错误"""
    pass


class SearchError(AutoAnalystError):
    """搜索错误"""
    pass


class TavilySearchError(SearchError):
    """Tavily 搜索错误"""
    pass


class RAGError(AutoAnalystError):
    """RAG 处理错误"""
    pass


class VectorStoreError(RAGError):
    """向量存储错误"""
    pass


class RerankError(RAGError):
    """重排序错误"""
    pass
