# tests/test_exceptions.py
"""异常模块单元测试"""
import pytest


class TestExceptions:
    """自定义异常测试"""

    def test_auto_analyst_error_is_base(self):
        """测试 AutoAnalystError 是基础异常"""
        from exceptions import AutoAnalystError

        error = AutoAnalystError("测试错误")
        assert isinstance(error, Exception)
        assert str(error) == "测试错误"

    def test_api_error_inheritance(self):
        """测试 APIError 继承自 AutoAnalystError"""
        from exceptions import APIError, AutoAnalystError

        error = APIError("API 错误")
        assert isinstance(error, AutoAnalystError)
        assert isinstance(error, Exception)

    def test_deepseek_api_error_inheritance(self):
        """测试 DeepSeekAPIError 继承链"""
        from exceptions import DeepSeekAPIError, APIError, AutoAnalystError

        error = DeepSeekAPIError("DeepSeek 错误")
        assert isinstance(error, APIError)
        assert isinstance(error, AutoAnalystError)

    def test_search_error_inheritance(self):
        """测试 SearchError 继承自 AutoAnalystError"""
        from exceptions import SearchError, AutoAnalystError

        error = SearchError("搜索错误")
        assert isinstance(error, AutoAnalystError)

    def test_tavily_search_error_inheritance(self):
        """测试 TavilySearchError 继承链"""
        from exceptions import TavilySearchError, SearchError, AutoAnalystError

        error = TavilySearchError("Tavily 错误")
        assert isinstance(error, SearchError)
        assert isinstance(error, AutoAnalystError)

    def test_rag_error_inheritance(self):
        """测试 RAGError 继承自 AutoAnalystError"""
        from exceptions import RAGError, AutoAnalystError

        error = RAGError("RAG 错误")
        assert isinstance(error, AutoAnalystError)

    def test_vector_store_error_inheritance(self):
        """测试 VectorStoreError 继承链"""
        from exceptions import VectorStoreError, RAGError, AutoAnalystError

        error = VectorStoreError("向量库错误")
        assert isinstance(error, RAGError)
        assert isinstance(error, AutoAnalystError)

    def test_rerank_error_inheritance(self):
        """测试 RerankError 继承链"""
        from exceptions import RerankError, RAGError, AutoAnalystError

        error = RerankError("重排序错误")
        assert isinstance(error, RAGError)
        assert isinstance(error, AutoAnalystError)

    def test_exception_can_be_caught_by_base(self):
        """测试子异常可以被基类捕获"""
        from exceptions import AutoAnalystError, DeepSeekAPIError

        with pytest.raises(AutoAnalystError):
            raise DeepSeekAPIError("测试")

    def test_exception_message_preserved(self):
        """测试异常消息被正确保留"""
        from exceptions import VectorStoreError

        message = "详细的错误消息"
        error = VectorStoreError(message)

        assert message in str(error)
        assert error.args[0] == message
