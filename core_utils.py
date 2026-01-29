# core_utils.py
"""
核心工具模块 - 封装 DeepSeek 大模型对话 和 Tavily 网络搜索

【作用】
提供两个核心能力：
1. chat_with_deepseek()：调用 DeepSeek 大模型进行对话（非流式）
2. chat_with_deepseek_stream()：调用 DeepSeek 大模型进行对话（流式，边生成边返回）
3. search_industry_info()：调用 Tavily 搜索引擎搜索行业信息

【关键概念】
- OpenAI SDK：DeepSeek 的 API 兼容 OpenAI 的接口格式，所以可以用 OpenAI 的 Python SDK 来调用
- Tavily：一个专为 AI 设计的搜索引擎 API，返回结构化的搜索结果
- 指数退避重试：失败后等待 1s → 2s → 4s 再重试，避免频繁请求
- Generator（生成器）：Python 的 yield 语法，用于流式输出，边产生数据边返回
"""
import time
from typing import List, Dict, Any, Generator
from openai import OpenAI        # OpenAI 官方 Python SDK，DeepSeek 兼容此接口
from tavily import TavilyClient  # Tavily 搜索引擎的 Python SDK

from config import Config, get_logger
from exceptions import DeepSeekAPIError, TavilySearchError

# 创建本模块专属的日志记录器
logger = get_logger(__name__)


class AnalystCore:
    """
    分析核心类 - 封装所有外部 API 调用

    【职责】
    - 与 DeepSeek 大模型通信（对话、流式对话）
    - 与 Tavily 搜索引擎通信（网络搜索）

    【为什么封装成类？】
    因为 OpenAI 和 Tavily 客户端初始化有开销（网络连接等），
    封装成类后只需初始化一次，后续复用同一个连接。
    """

    def __init__(self):
        """
        初始化 API 客户端

        创建两个客户端：
        1. self.client：OpenAI SDK 客户端（用于调用 DeepSeek）
        2. self.tavily：Tavily 搜索客户端
        """
        # 创建 OpenAI 兼容客户端，指向 DeepSeek 的 API 地址
        # 因为 DeepSeek 兼容 OpenAI API 格式，所以可以直接用 OpenAI SDK
        self.client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL   # 关键：把地址从 OpenAI 改为 DeepSeek
        )

        # 创建 Tavily 搜索客户端
        self.tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)
        logger.info("AnalystCore 初始化完成")

    def chat_with_deepseek(
        self,
        prompt: str,
        system_prompt: str = "You are a professional industry analyst."
    ) -> str:
        """
        调用 DeepSeek API 进行对话（非流式，等待完整响应）

        【工作流程】
        1. 发送 system_prompt（设定 AI 的角色）和 prompt（用户的问题）
        2. 等待 DeepSeek 生成完整回复
        3. 如果失败，使用指数退避策略重试（最多 3 次）

        【什么是指数退避？】
        第 1 次失败：等 1 秒（2^0）后重试
        第 2 次失败：等 2 秒（2^1）后重试
        第 3 次失败：抛出异常，不再重试

        Args:
            prompt: 用户的提示词（你想问 AI 什么）
            system_prompt: 系统提示词（告诉 AI 它是什么角色）

        Returns:
            AI 的完整响应文本

        Raises:
            DeepSeekAPIError: 所有重试都失败时抛出
        """
        last_error = None

        # 重试循环：最多尝试 MAX_RETRIES 次
        for i in range(Config.MAX_RETRIES):
            try:
                logger.debug(f"调用 DeepSeek API (尝试 {i + 1}/{Config.MAX_RETRIES})")

                # 调用 OpenAI 兼容的 Chat Completions API
                # messages 格式：[系统消息, 用户消息]
                response = self.client.chat.completions.create(
                    model=Config.DEEPSEEK_MODEL_NAME,  # 使用 "deepseek-chat" 模型
                    messages=[
                        {"role": "system", "content": system_prompt},  # 设定 AI 角色
                        {"role": "user", "content": prompt},           # 用户的问题
                    ],
                    stream=False  # 非流式：等待完整响应后一次性返回
                )

                logger.info("DeepSeek API 调用成功")
                # response.choices[0].message.content 就是 AI 的回复文本
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                logger.warning(f"DeepSeek API 调用失败 (尝试 {i + 1}): {e}")

                # 如果不是最后一次尝试，等待后重试
                if i < Config.MAX_RETRIES - 1:
                    sleep_time = 2 ** i  # 指数退避：1s, 2s, 4s...
                    logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)

        # 所有重试都失败，抛出自定义异常
        error_msg = f"DeepSeek API 调用失败，已重试 {Config.MAX_RETRIES} 次: {last_error}"
        logger.error(error_msg)
        raise DeepSeekAPIError(error_msg)

    def chat_with_deepseek_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a professional industry analyst."
    ) -> Generator[str, None, None]:
        """
        流式调用 DeepSeek API（边生成边返回，实现打字机效果）

        【与非流式的区别】
        - 非流式（stream=False）：等 AI 说完所有话，一次性返回整段文本
        - 流式（stream=True）：AI 每生成一个词就立刻返回，前端可以实时显示

        【什么是 Generator（生成器）？】
        使用 yield 关键字的函数叫生成器函数。
        它不会一次性返回所有结果，而是每次 yield 一个值，
        调用方通过 for 循环逐个获取：
            for chunk in core.chat_with_deepseek_stream(prompt):
                print(chunk, end="")  # 逐字打印

        Args:
            prompt: 用户的提示词
            system_prompt: 系统提示词

        Yields:
            每次返回一小段文本（通常是几个字）

        Raises:
            DeepSeekAPIError: API 调用失败时抛出
        """
        try:
            logger.debug("开始流式调用 DeepSeek API")

            # stream=True 让 API 返回一个可迭代的流（而不是完整响应）
            response = self.client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True  # 关键：开启流式模式
            )

            # 遍历流中的每个"块"（chunk）
            # 每个 chunk 包含一小段 AI 生成的文本
            for chunk in response:
                # chunk.choices[0].delta.content 是这个块的文本内容
                # 可能为 None（如流结束时），所以要判断
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content  # 用 yield 逐块返回

            logger.info("DeepSeek 流式调用完成")

        except Exception as e:
            error_msg = f"DeepSeek 流式调用失败: {e}"
            logger.error(error_msg)
            raise DeepSeekAPIError(error_msg)

    def search_industry_info(self, query: str) -> List[Dict[str, Any]]:
        """
        使用 Tavily 搜索引擎搜索行业信息

        【Tavily 是什么？】
        Tavily 是一个专为 AI 应用设计的搜索 API：
        - 返回结构化结果（标题、内容、URL），比直接爬网页方便
        - 支持 "advanced" 深度搜索，结果更全面
        - 用法类似百度/Google，但通过 API 调用

        【返回数据格式】
        [
            {"content": "文章内容...", "url": "https://...", "title": "标题"},
            {"content": "文章内容...", "url": "https://...", "title": "标题"},
        ]

        Args:
            query: 搜索关键词（如 "2024年中国人形机器人产业"）

        Returns:
            搜索结果列表，每个结果是一个字典

        Raises:
            TavilySearchError: 搜索失败时抛出
        """
        try:
            # 日志中只记录前 50 个字符，避免超长查询刷屏
            logger.info(f"执行 Tavily 搜索: {query[:50]}...")

            # 调用 Tavily API 执行搜索
            search_result = self.tavily.search(
                query=query,
                search_depth=Config.TAVILY_SEARCH_DEPTH,  # "advanced" 深度搜索
                max_results=Config.TAVILY_MAX_RESULTS      # 最多返回 5 条
            )

            # 从响应中提取 results 列表（如果没有则返回空列表）
            results = search_result.get("results", [])
            logger.info(f"搜索完成，获取 {len(results)} 条结果")
            return results

        except Exception as e:
            error_msg = f"Tavily 搜索失败: {e}"
            logger.error(error_msg)
            raise TavilySearchError(error_msg)
