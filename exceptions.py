# exceptions.py
"""
自定义异常模块

【作用】
定义项目专属的异常类，替代通用的 Exception。
好处：
1. 可以精确捕获特定类型的错误（如只捕获 API 错误，不捕获其他错误）
2. 面试时体现"异常分层设计"的工程素养
3. 日志中能清晰看到是哪一层出了问题

【异常继承关系】（从上到下，父类 → 子类）

AutoAnalystError（基类，所有自定义异常的父类）
├── ConfigError          - 配置错误
├── APIError             - API 调用错误
│   └── DeepSeekAPIError     - DeepSeek API 专属错误
├── SearchError          - 搜索错误
│   └── TavilySearchError    - Tavily 搜索专属错误
└── RAGError             - RAG 处理错误
    ├── VectorStoreError     - 向量数据库错误（ChromaDB）
    └── RerankError          - 重排序错误（FlashRank）

【使用示例】
  try:
      result = core.search_industry_info(query)
  except TavilySearchError as e:
      # 只捕获 Tavily 搜索错误
      print(f"搜索失败: {e}")
  except AutoAnalystError as e:
      # 捕获所有自定义错误（兜底）
      print(f"系统错误: {e}")
"""


class AutoAnalystError(Exception):
    """
    Auto-Analyst 基础异常类

    所有自定义异常都继承自这个类。
    在最外层可以用 except AutoAnalystError 捕获所有业务异常。
    """
    pass


class ConfigError(AutoAnalystError):
    """配置错误 - 如缺少 API Key、配置格式错误等"""
    pass


class APIError(AutoAnalystError):
    """API 调用错误 - 所有外部 API 调用失败的基类"""
    pass


class DeepSeekAPIError(APIError):
    """
    DeepSeek API 调用错误

    触发场景：
    - 网络超时
    - API Key 无效
    - 模型调用失败
    - 重试次数耗尽
    """
    pass


class SearchError(AutoAnalystError):
    """搜索错误 - 所有搜索相关失败的基类"""
    pass


class TavilySearchError(SearchError):
    """
    Tavily 搜索错误

    触发场景：
    - Tavily API 调用失败
    - 搜索超时
    - API Key 无效
    """
    pass


class RAGError(AutoAnalystError):
    """RAG 处理错误 - 检索增强生成流程中的错误基类"""
    pass


class VectorStoreError(RAGError):
    """
    向量存储错误

    触发场景：
    - ChromaDB 初始化失败
    - 文档写入失败
    - 向量检索失败
    - 清空数据库失败
    """
    pass


class RerankError(RAGError):
    """
    重排序错误

    触发场景：
    - FlashRank 模型加载失败
    - 重排序执行失败
    """
    pass
