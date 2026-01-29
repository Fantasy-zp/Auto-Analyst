# rag_processor.py
"""
RAG 处理模块 - 实现两阶段检索（向量搜索 + 重排序）

【RAG 是什么？】
RAG = Retrieval-Augmented Generation（检索增强生成）
核心思想：先从知识库中"检索"相关信息，再把这些信息作为"上下文"交给大模型生成回答。
这样大模型就不是凭空编造，而是基于真实数据回答。

【为什么需要两阶段？】
- 第一阶段（向量搜索）：快速但粗糙 - 从大量文档中快速找出"可能相关"的候选（召回 10 条）
- 第二阶段（重排序）：慢但精准 - 用专门的模型对候选重新打分，只保留最相关的（筛选 3 条）

【数据流】
Tavily 搜索结果 → 存入 ChromaDB 向量库 → 向量检索召回 10 条 → FlashRank 重排序 → 返回 Top 3

【关键技术】
- ChromaDB：轻量级向量数据库，把文本转为向量存储，支持相似度搜索
- 向量/Embedding：把文本转换为一组数字（如 384 维向量），语义相似的文本向量也相近
- FlashRank：轻量级重排序模型，CPU 即可运行，用于对检索结果精细排序
- 重排序 vs 向量搜索：向量搜索基于"语义相似度"，重排序基于"查询-文档相关性"，更精准
"""
import hashlib
from typing import List, Dict, Any
import chromadb                                     # 向量数据库
from chromadb.utils import embedding_functions      # 文本转向量的工具
from flashrank import Ranker, RerankRequest         # 重排序模型

from config import Config, get_logger
from exceptions import VectorStoreError, RerankError

logger = get_logger(__name__)


class AdvancedRAG:
    """
    高级 RAG 实现类 - 两阶段检索（向量搜索 + 重排序）

    【核心流程】
    1. add_documents()：把搜索到的网页内容存入向量数据库
    2. retrieve_and_rerank()：根据查询从向量库检索，再用重排序模型精选
    3. clear_db()：清空向量数据库

    【类比理解】
    想象你在图书馆找书：
    - 向量搜索 = 先到"计算机"分区找出所有相关的书（快速但粗糙）
    - 重排序 = 从这些书中，根据你的具体需求挑出最匹配的 3 本（精准）
    """

    def __init__(self):
        """
        初始化三个核心组件：
        1. ChromaDB 向量数据库客户端
        2. 文本嵌入函数（把文字变成向量）
        3. FlashRank 重排序模型
        """
        try:
            # ---------- 组件 1：向量数据库 ----------
            # PersistentClient = 持久化客户端，数据保存在本地磁盘
            # 即使程序重启，之前存入的数据还在
            logger.info(f"初始化 ChromaDB，路径: {Config.CHROMA_DB_PATH}")
            self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)

            # ---------- 组件 2：嵌入函数（Embedding Function）----------
            # 嵌入函数的作用：把一段文字 → 一组数字（向量）
            # 例如："人形机器人" → [0.12, -0.34, 0.56, ..., 0.78]（384 个数字）
            # 语义相似的文字，转换后的向量也会很接近
            # DefaultEmbeddingFunction 使用 all-MiniLM-L6-v2 模型，在本地 CPU 运行
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

            # ---------- 组件 3：重排序模型 ----------
            # FlashRank 是一个轻量级的重排序模型
            # 作用：给定查询 Q 和一组文档 D，判断每个文档与查询的相关程度
            # 相比向量搜索的"语义相似度"，重排序更擅长判断"这个文档是否真的回答了这个问题"
            logger.info(f"初始化 FlashRank Reranker: {Config.RAG_RERANK_MODEL}")
            self.ranker = Ranker(
                model_name=Config.RAG_RERANK_MODEL,    # ms-marco-MiniLM-L-12-v2
                cache_dir=Config.FLASHRANK_CACHE_PATH   # 模型文件缓存到 ./opt 目录
            )

            # ---------- 组件 4：集合（Collection）----------
            # Collection 类似数据库中的"表"，存放向量化的文档
            # get_or_create：如果已有则直接使用，没有则新建
            self.collection = self.chroma_client.get_or_create_collection(
                name=Config.RAG_COLLECTION_NAME,        # 集合名："industry_reports"
                embedding_function=self.embedding_fn     # 告诉集合用什么方式把文字转向量
            )
            logger.info("AdvancedRAG 初始化完成")

        except Exception as e:
            error_msg = f"AdvancedRAG 初始化失败: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        把搜索结果存入向量数据库

        【流程】
        1. 为每个文档生成唯一 ID（基于内容哈希，同内容不会重复存储）
        2. 过滤掉空内容的文档
        3. 调用 ChromaDB 的 add 方法存入

        【存入时发生了什么？】
        ChromaDB 会自动调用 embedding_fn 把每个文档的文本转为向量，
        然后把"向量 + 原文 + 元数据"一起存储。

        Args:
            documents: 文档列表，每个文档格式：
                {"content": "文章内容...", "url": "https://来源网址"}

        Raises:
            VectorStoreError: 存储失败时抛出
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return

        try:
            # 基于文档内容生成 MD5 哈希作为 ID
            # 好处：同一篇文章多次存入时，ID 相同，ChromaDB 会自动去重
            # 例如："人形机器人..." → "a1b2c3d4e5f6..." (MD5 哈希值)
            ids = [
                hashlib.md5(d.get("content", "").encode()).hexdigest()
                for d in documents
            ]
            # 元数据：记录每个文档的来源 URL
            metadatas = [{"source": d.get("url", "local")} for d in documents]
            # 提取文档正文内容
            contents = [d.get("content", "") for d in documents]

            # 过滤空内容：zip 三个列表，只保留 content 非空的条目
            valid_data = [
                (id_, meta, content)
                for id_, meta, content in zip(ids, metadatas, contents)
                if content.strip()  # strip() 去除首尾空格后判断是否为空
            ]

            if not valid_data:
                logger.warning("所有文档内容为空，跳过添加")
                return

            # zip(*valid_data) 是 zip 的逆操作：把三元组列表拆回三个独立列表
            valid_ids, valid_metadatas, valid_contents = zip(*valid_data)

            # 存入 ChromaDB
            # ChromaDB 会自动：1. 用 embedding_fn 把文本转向量 2. 存储向量+原文+元数据
            self.collection.add(
                ids=list(valid_ids),
                metadatas=list(valid_metadatas),
                documents=list(valid_contents)
            )
            logger.info(f"成功添加 {len(valid_ids)} 条文档到向量库")

        except Exception as e:
            error_msg = f"添加文档到向量库失败: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)

    def retrieve_and_rerank(self, query: str, top_k: int = None) -> str:
        """
        核心流程：两阶段检索（向量搜索 → 重排序）

        【第一阶段：向量搜索（粗筛）】
        把查询文本转为向量，在向量库中找出最相似的 10 条文档
        原理：余弦相似度 - 两个向量的方向越接近，文本语义越相似

        【第二阶段：重排序（精筛）】
        用 FlashRank 模型对这 10 条文档重新打分
        原理：交叉编码器 - 同时看"查询"和"文档"，判断相关性（比向量搜索更精准）

        【举例】
        查询："人形机器人市场前景"
        向量搜索可能返回：[机器人技术, 天气预报, 机器人市场, 做菜教程, ...]
        重排序会重新排列：[机器人市场(0.95), 机器人技术(0.87), 天气预报(0.12), ...]
        只取 Top 3：[机器人市场, 机器人技术, ...]

        Args:
            query: 查询文本
            top_k: 最终返回多少条结果（默认 3）

        Returns:
            拼接后的精选上下文文本，用 "---" 分隔

        Raises:
            RerankError: 检索或重排序失败时抛出
        """
        if top_k is None:
            top_k = Config.RAG_RERANK_TOP_K  # 默认取 3 条

        try:
            # ============ 第一阶段：向量搜索 ============
            logger.info(f"执行向量检索，召回 {Config.RAG_RETRIEVE_COUNT} 条结果")

            # collection.query() 执行向量相似度搜索
            # 内部流程：query 文本 → embedding_fn 转向量 → 在所有向量中找最近的 N 个
            results = self.collection.query(
                query_texts=[query],                    # 查询文本（会自动转为向量）
                n_results=Config.RAG_RETRIEVE_COUNT     # 召回 10 条候选
            )

            # 安全检查：确保返回了有效结果
            if not results or 'documents' not in results:
                logger.warning("向量检索返回空结果")
                return "未找到相关背景资料。"

            documents = results['documents']
            # results['documents'] 的格式是 [[doc1, doc2, ...]]（二维列表）
            # 因为 query_texts 传了 1 个查询，所以取 [0] 获取第一个查询的结果
            if not documents or not documents[0]:
                logger.warning("向量检索未找到相关文档")
                return "未找到相关背景资料。"

            docs = documents[0]  # 获取第一个查询的结果列表
            logger.info(f"向量检索返回 {len(docs)} 条文档")

            # ============ 第二阶段：重排序 ============
            logger.info(f"执行重排序，筛选 top {top_k} 条")

            # 把文档列表转换为 FlashRank 需要的格式
            passages = [{"id": i, "text": doc} for i, doc in enumerate(docs)]

            # 构建重排序请求：查询 + 候选文档列表
            rerank_request = RerankRequest(query=query, passages=passages)

            # 执行重排序：FlashRank 会为每个文档计算与查询的相关性得分
            # 返回结果已按得分从高到低排序
            rerank_results = self.ranker.rerank(rerank_request)

            # 只取得分最高的 top_k 条
            top_passages = rerank_results[:top_k]

            # 把精选文档的文本拼接成一段上下文，用 "---" 分隔
            context_list = [p['text'] for p in top_passages]
            final_context = "\n\n---\n\n".join(context_list)

            logger.info(f"重排序完成，返回 {len(top_passages)} 条精选结果")
            return final_context

        except Exception as e:
            error_msg = f"检索和重排序失败: {e}"
            logger.error(error_msg)
            raise RerankError(error_msg)

    def clear_db(self) -> None:
        """
        清空向量数据库中的所有数据

        【流程】
        1. 删除整个集合
        2. 重新创建空集合

        【使用场景】
        - 切换研究主题时，避免旧数据干扰新查询
        - 数据库积累过多导致检索不精准时

        Raises:
            VectorStoreError: 清理失败时抛出
        """
        try:
            logger.info("清理向量数据库...")
            # 删除整个集合（包括其中的所有向量和文档）
            self.chroma_client.delete_collection(Config.RAG_COLLECTION_NAME)
            # 重新创建空集合
            self.collection = self.chroma_client.create_collection(
                name=Config.RAG_COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            logger.info("向量数据库已清空")
        except Exception as e:
            error_msg = f"清理向量数据库失败: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
