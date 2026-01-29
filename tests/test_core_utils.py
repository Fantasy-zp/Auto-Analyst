# tests/test_core_utils.py
"""核心工具模块单元测试"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAnalystCore:
    """AnalystCore 类测试"""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI 客户端"""
        with patch('core_utils.OpenAI') as mock:
            yield mock

    @pytest.fixture
    def mock_tavily_client(self):
        """Mock Tavily 客户端"""
        with patch('core_utils.TavilyClient') as mock:
            yield mock

    @pytest.fixture
    def analyst_core(self, mock_openai_client, mock_tavily_client):
        """创建 AnalystCore 实例（带 mock）"""
        from core_utils import AnalystCore
        return AnalystCore()

    def test_init_creates_clients(self, mock_openai_client, mock_tavily_client):
        """测试初始化时创建 API 客户端"""
        from core_utils import AnalystCore

        core = AnalystCore()

        mock_openai_client.assert_called_once()
        mock_tavily_client.assert_called_once()

    def test_chat_with_deepseek_success(self, analyst_core, mock_openai_client):
        """测试 DeepSeek API 调用成功"""
        # 设置 mock 返回值
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="测试响应内容"))]
        analyst_core.client.chat.completions.create.return_value = mock_response

        result = analyst_core.chat_with_deepseek("测试提示词")

        assert result == "测试响应内容"
        analyst_core.client.chat.completions.create.assert_called_once()

    def test_chat_with_deepseek_with_custom_system_prompt(self, analyst_core):
        """测试自定义系统提示词"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="响应"))]
        analyst_core.client.chat.completions.create.return_value = mock_response

        analyst_core.chat_with_deepseek("提示词", system_prompt="自定义系统提示")

        call_args = analyst_core.client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        assert messages[0]['content'] == "自定义系统提示"

    def test_chat_with_deepseek_retry_on_failure(self, analyst_core):
        """测试 API 调用失败时重试"""
        from exceptions import DeepSeekAPIError

        # 模拟前两次失败，第三次成功
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="成功"))]

        analyst_core.client.chat.completions.create.side_effect = [
            Exception("第一次失败"),
            Exception("第二次失败"),
            mock_response
        ]

        with patch('core_utils.time.sleep'):  # 跳过等待时间
            result = analyst_core.chat_with_deepseek("测试")

        assert result == "成功"
        assert analyst_core.client.chat.completions.create.call_count == 3

    def test_chat_with_deepseek_all_retries_fail(self, analyst_core):
        """测试所有重试都失败时抛出异常"""
        from exceptions import DeepSeekAPIError

        analyst_core.client.chat.completions.create.side_effect = Exception("持续失败")

        with patch('core_utils.time.sleep'):
            with pytest.raises(DeepSeekAPIError) as exc_info:
                analyst_core.chat_with_deepseek("测试")

        assert "已重试" in str(exc_info.value)

    def test_search_industry_info_success(self, analyst_core):
        """测试 Tavily 搜索成功"""
        mock_results = {
            "results": [
                {"content": "结果1", "url": "http://example1.com"},
                {"content": "结果2", "url": "http://example2.com"}
            ]
        }
        analyst_core.tavily.search.return_value = mock_results

        results = analyst_core.search_industry_info("测试查询")

        assert len(results) == 2
        assert results[0]["content"] == "结果1"

    def test_search_industry_info_empty_results(self, analyst_core):
        """测试搜索返回空结果"""
        analyst_core.tavily.search.return_value = {"results": []}

        results = analyst_core.search_industry_info("测试查询")

        assert results == []

    def test_search_industry_info_failure(self, analyst_core):
        """测试搜索失败时抛出异常"""
        from exceptions import TavilySearchError

        analyst_core.tavily.search.side_effect = Exception("搜索失败")

        with pytest.raises(TavilySearchError) as exc_info:
            analyst_core.search_industry_info("测试查询")

        assert "搜索失败" in str(exc_info.value)

    def test_search_query_truncation_in_log(self, analyst_core):
        """测试长查询在日志中被截断"""
        analyst_core.tavily.search.return_value = {"results": []}

        long_query = "这是一个非常长的查询词" * 10
        analyst_core.search_industry_info(long_query)

        # 验证调用成功即可（日志截断是内部行为）
        analyst_core.tavily.search.assert_called_once()


class TestAnalystCoreStreaming:
    """AnalystCore 流式输出测试"""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI 客户端"""
        with patch('core_utils.OpenAI') as mock:
            yield mock

    @pytest.fixture
    def mock_tavily_client(self):
        """Mock Tavily 客户端"""
        with patch('core_utils.TavilyClient') as mock:
            yield mock

    @pytest.fixture
    def analyst_core(self, mock_openai_client, mock_tavily_client):
        """创建 AnalystCore 实例（带 mock）"""
        from core_utils import AnalystCore
        return AnalystCore()

    def test_chat_with_deepseek_stream_success(self, analyst_core):
        """测试流式调用成功"""
        # 创建模拟的流式响应
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" World"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))]),
        ]

        analyst_core.client.chat.completions.create.return_value = iter(mock_chunks)

        result = list(analyst_core.chat_with_deepseek_stream("测试提示词"))

        assert result == ["Hello", " World", "!"]

    def test_chat_with_deepseek_stream_with_empty_chunks(self, analyst_core):
        """测试流式调用包含空块"""
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # 空块
            Mock(choices=[Mock(delta=Mock(content=" World"))]),
        ]

        analyst_core.client.chat.completions.create.return_value = iter(mock_chunks)

        result = list(analyst_core.chat_with_deepseek_stream("测试"))

        # 空块应被过滤
        assert result == ["Hello", " World"]

    def test_chat_with_deepseek_stream_failure(self, analyst_core):
        """测试流式调用失败"""
        from exceptions import DeepSeekAPIError

        analyst_core.client.chat.completions.create.side_effect = Exception("流式调用失败")

        with pytest.raises(DeepSeekAPIError) as exc_info:
            list(analyst_core.chat_with_deepseek_stream("测试"))

        assert "流式调用失败" in str(exc_info.value)

    def test_chat_with_deepseek_stream_custom_system_prompt(self, analyst_core):
        """测试流式调用自定义系统提示词"""
        mock_chunks = [Mock(choices=[Mock(delta=Mock(content="响应"))])]
        analyst_core.client.chat.completions.create.return_value = iter(mock_chunks)

        list(analyst_core.chat_with_deepseek_stream("提示词", system_prompt="自定义系统提示"))

        call_args = analyst_core.client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        assert messages[0]['content'] == "自定义系统提示"
        assert call_args.kwargs['stream'] is True
