# core_utils.py
"""核心工具模块 - 提供 DeepSeek chat 和 Tavily 搜索 API"""
import time
from typing import List, Dict, Any
from openai import OpenAI
from tavily import TavilyClient

from config import Config, get_logger
from exceptions import DeepSeekAPIError, TavilySearchError

logger = get_logger(__name__)


class AnalystCore:
    """分析核心类，封装 DeepSeek 和 Tavily API"""

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)
        logger.info("AnalystCore 初始化完成")

    def chat_with_deepseek(
        self,
        prompt: str,
        system_prompt: str = "You are a professional industry analyst."
    ) -> str:
        """
        调用 DeepSeek API 进行对话

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Returns:
            模型响应内容

        Raises:
            DeepSeekAPIError: API 调用失败时抛出
        """
        last_error = None
        for i in range(Config.MAX_RETRIES):
            try:
                logger.debug(f"调用 DeepSeek API (尝试 {i + 1}/{Config.MAX_RETRIES})")
                response = self.client.chat.completions.create(
                    model=Config.DEEPSEEK_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )
                logger.info("DeepSeek API 调用成功")
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(f"DeepSeek API 调用失败 (尝试 {i + 1}): {e}")
                if i < Config.MAX_RETRIES - 1:
                    sleep_time = 2 ** i
                    logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)

        error_msg = f"DeepSeek API 调用失败，已重试 {Config.MAX_RETRIES} 次: {last_error}"
        logger.error(error_msg)
        raise DeepSeekAPIError(error_msg)

    def search_industry_info(self, query: str) -> List[Dict[str, Any]]:
        """
        使用 Tavily 搜索行业信息

        Args:
            query: 搜索查询词

        Returns:
            搜索结果列表

        Raises:
            TavilySearchError: 搜索失败时抛出
        """
        try:
            logger.info(f"执行 Tavily 搜索: {query[:50]}...")
            search_result = self.tavily.search(
                query=query,
                search_depth=Config.TAVILY_SEARCH_DEPTH,
                max_results=Config.TAVILY_MAX_RESULTS
            )
            results = search_result.get("results", [])
            logger.info(f"搜索完成，获取 {len(results)} 条结果")
            return results
        except Exception as e:
            error_msg = f"Tavily 搜索失败: {e}"
            logger.error(error_msg)
            raise TavilySearchError(error_msg)


# 测试代码 (仅在直接运行此文件时执行)
if __name__ == "__main__":
    if not Config.validate():
        print("配置验证失败，请检查 .env 文件")
        exit(1)

    analyst = AnalystCore()

    test_query = "2024年中国人形机器人产业最新进展"
    print(f"--- 正在搜索: {test_query} ---")
    try:
        results = analyst.search_industry_info(test_query)
        context = "\n".join([f"来源: {r['url']}\n内容: {r['content']}" for r in results])
        summary_prompt = f"请根据以下搜索结果，总结关于'{test_query}'的核心观点：\n\n{context}"

        print("--- 正在调用 DeepSeek 进行总结 ---")
        summary = analyst.chat_with_deepseek(summary_prompt)
        print(f"\n总结结果：\n{summary}")
    except Exception as e:
        print(f"执行失败: {e}")
