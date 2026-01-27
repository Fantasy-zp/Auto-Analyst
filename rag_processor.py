# rag_processor.py
"""RAG 处理模块 - 实现两阶段检索（向量搜索 + 重排序）"""
import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest

from config import Config, get_logger
from exceptions import VectorStoreError, RerankError

logger = get_logger(__name__)


class AdvancedRAG:
    """高级 RAG 实现类，包含向量检索和重排序功能"""

    def __init__(self):
        try:
            # 1. 初始化本地向量库 (持久化在本地目录)
            logger.info(f"初始化 ChromaDB，路径: {Config.CHROMA_DB_PATH}")
            self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)

            # 2. 使用默认的嵌入函数
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

            # 3. 初始化 Reranker (FlashRank 轻量级模型，CPU 运行)
            logger.info(f"初始化 FlashRank Reranker: {Config.RAG_RERANK_MODEL}")
            self.ranker = Ranker(
                model_name=Config.RAG_RERANK_MODEL,
                cache_dir=Config.FLASHRANK_CACHE_PATH
            )

            # 4. 创建或获取集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=Config.RAG_COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            logger.info("AdvancedRAG 初始化完成")
        except Exception as e:
            error_msg = f"AdvancedRAG 初始化失败: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        将搜索结果或 PDF 片段存入向量库

        Args:
            documents: 文档列表，每个文档包含 content 和可选的 url

        Raises:
            VectorStoreError: 存储失败时抛出
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return

        try:
            ids = [f"id_{i}_{os.urandom(4).hex()}" for i in range(len(documents))]
            metadatas = [{"source": d.get("url", "local")} for d in documents]
            contents = [d.get("content", "") for d in documents]

            # 过滤空内容
            valid_data = [
                (id_, meta, content)
                for id_, meta, content in zip(ids, metadatas, contents)
                if content.strip()
            ]

            if not valid_data:
                logger.warning("所有文档内容为空，跳过添加")
                return

            valid_ids, valid_metadatas, valid_contents = zip(*valid_data)

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
        核心流程：检索 + 重排序

        Args:
            query: 查询文本
            top_k: 返回的文档数量，默认使用配置值

        Returns:
            拼接后的上下文文本

        Raises:
            RAGError: 检索或重排序失败时抛出
        """
        if top_k is None:
            top_k = Config.RAG_RERANK_TOP_K

        try:
            # --- 第一阶段：初筛 (Vector Search) ---
            logger.info(f"执行向量检索，召回 {Config.RAG_RETRIEVE_COUNT} 条结果")
            results = self.collection.query(
                query_texts=[query],
                n_results=Config.RAG_RETRIEVE_COUNT
            )

            # 安全检查 documents 是否存在且不为空
            if not results or 'documents' not in results:
                logger.warning("向量检索返回空结果")
                return "未找到相关背景资料。"

            documents = results['documents']
            if not documents or not documents[0]:
                logger.warning("向量检索未找到相关文档")
                return "未找到相关背景资料。"

            docs = documents[0]
            logger.info(f"向量检索返回 {len(docs)} 条文档")

            # --- 第二阶段：精筛 (Reranking) ---
            logger.info(f"执行重排序，筛选 top {top_k} 条")
            passages = [{"id": i, "text": doc} for i, doc in enumerate(docs)]
            rerank_request = RerankRequest(query=query, passages=passages)

            # 执行重排序
            rerank_results = self.ranker.rerank(rerank_request)

            # 只取得分最高的 top_k 条
            top_passages = rerank_results[:top_k]

            # 拼接最终的上下文
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
        清理旧数据，防止干扰

        Raises:
            VectorStoreError: 清理失败时抛出
        """
        try:
            logger.info("清理向量数据库...")
            self.chroma_client.delete_collection(Config.RAG_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=Config.RAG_COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            logger.info("向量数据库已清空")
        except Exception as e:
            error_msg = f"清理向量数据库失败: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)


# 测试代码
if __name__ == "__main__":
    rag = AdvancedRAG()

    # 模拟一些脏数据和干扰数据
    mock_data = [
        {"content": "人形机器人通过液压驱动实现复杂动作，波士顿动力是行业标杆。", "url": "tech_news_1"},
        {"content": "今天天气不错，适合出去郊游。", "url": "daily_blog"},  # 干扰数据
        {"content": "2024年机器人市场预计增长20%，核心部件依赖进口。", "url": "market_report_1"},
        {"content": "如何在家制作一份完美的宫保鸡丁。", "url": "cook_book"}  # 干扰数据
    ]

    rag.add_documents(mock_data)

    print("--- 正在进行检索与重排序 ---")
    query = "人形机器人的技术发展和市场前景"
    context = rag.retrieve_and_rerank(query)

    print("\n[最终选取的上下文]:")
    print(context)
