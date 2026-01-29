# tests/conftest.py
"""Pytest 配置和共享 fixtures"""
import os
import sys
import pytest

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def setup_test_env():
    """为所有测试设置测试环境"""
    # 设置测试用的环境变量
    os.environ.setdefault("DEEPSEEK_API_KEY", "test_deepseek_key")
    os.environ.setdefault("TAVILY_API_KEY", "test_tavily_key")

    yield

    # 测试后清理（如果需要）


@pytest.fixture
def sample_documents():
    """提供测试用的示例文档"""
    return [
        {
            "content": "人形机器人技术在2024年取得重大突破。",
            "url": "https://example.com/robots"
        },
        {
            "content": "人工智能行业市场规模持续增长。",
            "url": "https://example.com/ai-market"
        },
        {
            "content": "新能源汽车销量创历史新高。",
            "url": "https://example.com/ev"
        }
    ]


@pytest.fixture
def sample_search_results():
    """提供测试用的搜索结果"""
    return {
        "results": [
            {
                "title": "机器人技术发展报告",
                "content": "2024年机器人行业发展迅速...",
                "url": "https://example.com/report1"
            },
            {
                "title": "AI市场分析",
                "content": "人工智能市场预计增长30%...",
                "url": "https://example.com/report2"
            }
        ]
    }


@pytest.fixture
def mock_llm_response():
    """提供模拟的 LLM 响应"""
    return "这是一份关于人形机器人行业的深度分析报告..."
