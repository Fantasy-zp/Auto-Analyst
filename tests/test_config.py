# tests/test_config.py
"""配置模块单元测试"""
import os
import pytest
from unittest.mock import patch


class TestConfig:
    """Config 类测试"""

    def test_config_loads_env_variables(self):
        """测试配置能正确加载环境变量"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test_deepseek_key",
            "TAVILY_API_KEY": "test_tavily_key"
        }):
            # 重新导入以获取新的环境变量
            import importlib
            import config
            importlib.reload(config)

            assert config.Config.DEEPSEEK_API_KEY == "test_deepseek_key"
            assert config.Config.TAVILY_API_KEY == "test_tavily_key"

    def test_config_default_values(self):
        """测试配置默认值"""
        from config import Config

        assert Config.DEEPSEEK_BASE_URL == "https://api.deepseek.com"
        assert Config.DEEPSEEK_MODEL_NAME == "deepseek-chat"
        assert Config.DEEPSEEK_MODEL == "deepseek/deepseek-chat"
        assert Config.RAG_RETRIEVE_COUNT == 10
        assert Config.RAG_RERANK_TOP_K == 3
        assert Config.MAX_RETRIES == 3

    def test_validate_with_missing_keys(self):
        """测试缺少 API Key 时验证失败"""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import config
            importlib.reload(config)

            # 手动设置为 None 模拟缺失
            config.Config.DEEPSEEK_API_KEY = None
            config.Config.TAVILY_API_KEY = None

            assert config.Config.validate() is False

    def test_validate_with_all_keys(self):
        """测试所有 API Key 存在时验证成功"""
        from config import Config

        # 临时设置 API Key
        original_deepseek = Config.DEEPSEEK_API_KEY
        original_tavily = Config.TAVILY_API_KEY

        Config.DEEPSEEK_API_KEY = "test_key"
        Config.TAVILY_API_KEY = "test_key"

        try:
            assert Config.validate() is True
        finally:
            # 恢复原值
            Config.DEEPSEEK_API_KEY = original_deepseek
            Config.TAVILY_API_KEY = original_tavily

    def test_setup_openai_env(self):
        """测试 OpenAI 环境变量设置"""
        from config import Config

        original_key = Config.DEEPSEEK_API_KEY
        Config.DEEPSEEK_API_KEY = "test_api_key"

        try:
            Config.setup_openai_env()

            assert os.environ.get("DEEPSEEK_API_KEY") == "test_api_key"
            assert os.environ.get("OPENAI_API_KEY") == "test_api_key"
            assert os.environ.get("OTEL_SDK_DISABLED") == "true"
        finally:
            Config.DEEPSEEK_API_KEY = original_key


class TestGetLogger:
    """get_logger 函数测试"""

    def test_get_logger_returns_logger(self):
        """测试 get_logger 返回正确的 logger"""
        from config import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_get_logger_different_names(self):
        """测试不同名称返回不同 logger"""
        from config import get_logger

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name != logger2.name
