# tests/test_rag_processor.py
"""RAG 处理模块单元测试"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAdvancedRAG:
    """AdvancedRAG 类测试"""

    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB"""
        with patch('rag_processor.chromadb.PersistentClient') as mock:
            mock_collection = Mock()
            mock.return_value.get_or_create_collection.return_value = mock_collection
            mock.return_value.create_collection.return_value = mock_collection
            yield mock

    @pytest.fixture
    def mock_embedding(self):
        """Mock 嵌入函数"""
        with patch('rag_processor.embedding_functions.DefaultEmbeddingFunction') as mock:
            yield mock

    @pytest.fixture
    def mock_ranker(self):
        """Mock FlashRank Ranker"""
        with patch('rag_processor.Ranker') as mock:
            yield mock

    @pytest.fixture
    def rag_instance(self, mock_chromadb, mock_embedding, mock_ranker):
        """创建 AdvancedRAG 实例（带 mock）"""
        from rag_processor import AdvancedRAG
        return AdvancedRAG()

    def test_init_creates_components(self, mock_chromadb, mock_embedding, mock_ranker):
        """测试初始化时创建所有组件"""
        from rag_processor import AdvancedRAG

        rag = AdvancedRAG()

        mock_chromadb.assert_called_once()
        mock_embedding.assert_called_once()
        mock_ranker.assert_called_once()

    def test_init_failure_raises_error(self, mock_chromadb, mock_embedding, mock_ranker):
        """测试初始化失败时抛出 VectorStoreError"""
        from exceptions import VectorStoreError

        mock_chromadb.side_effect = Exception("数据库连接失败")

        with pytest.raises(VectorStoreError) as exc_info:
            from rag_processor import AdvancedRAG
            AdvancedRAG()

        assert "初始化失败" in str(exc_info.value)

    def test_add_documents_success(self, rag_instance):
        """测试添加文档成功"""
        documents = [
            {"content": "文档1内容", "url": "http://example1.com"},
            {"content": "文档2内容", "url": "http://example2.com"}
        ]

        rag_instance.add_documents(documents)

        rag_instance.collection.add.assert_called_once()
        call_args = rag_instance.collection.add.call_args
        assert len(call_args.kwargs['documents']) == 2

    def test_add_documents_empty_list(self, rag_instance):
        """测试添加空文档列表"""
        rag_instance.add_documents([])

        rag_instance.collection.add.assert_not_called()

    def test_add_documents_filters_empty_content(self, rag_instance):
        """测试过滤空内容文档"""
        documents = [
            {"content": "有效内容", "url": "http://example.com"},
            {"content": "", "url": "http://empty.com"},
            {"content": "   ", "url": "http://whitespace.com"}
        ]

        rag_instance.add_documents(documents)

        call_args = rag_instance.collection.add.call_args
        assert len(call_args.kwargs['documents']) == 1
        assert call_args.kwargs['documents'][0] == "有效内容"

    def test_add_documents_all_empty(self, rag_instance):
        """测试所有文档内容为空时不添加"""
        documents = [
            {"content": "", "url": "http://empty1.com"},
            {"content": "  ", "url": "http://empty2.com"}
        ]

        rag_instance.add_documents(documents)

        rag_instance.collection.add.assert_not_called()

    def test_add_documents_failure(self, rag_instance):
        """测试添加文档失败时抛出异常"""
        from exceptions import VectorStoreError

        rag_instance.collection.add.side_effect = Exception("存储失败")

        with pytest.raises(VectorStoreError) as exc_info:
            rag_instance.add_documents([{"content": "测试", "url": "test"}])

        assert "添加文档" in str(exc_info.value)

    def test_retrieve_and_rerank_success(self, rag_instance, mock_ranker):
        """测试检索和重排序成功"""
        # Mock 向量检索结果
        rag_instance.collection.query.return_value = {
            'documents': [['文档1', '文档2', '文档3']]
        }

        # Mock 重排序结果
        rag_instance.ranker.rerank.return_value = [
            {'text': '文档1', 'score': 0.9},
            {'text': '文档2', 'score': 0.8}
        ]

        result = rag_instance.retrieve_and_rerank("测试查询", top_k=2)

        assert "文档1" in result
        assert "文档2" in result
        rag_instance.collection.query.assert_called_once()
        rag_instance.ranker.rerank.assert_called_once()

    def test_retrieve_and_rerank_empty_results(self, rag_instance):
        """测试向量检索返回空结果"""
        rag_instance.collection.query.return_value = {'documents': [[]]}

        result = rag_instance.retrieve_and_rerank("测试查询")

        assert "未找到" in result

    def test_retrieve_and_rerank_no_documents_key(self, rag_instance):
        """测试返回结果缺少 documents 键"""
        rag_instance.collection.query.return_value = {}

        result = rag_instance.retrieve_and_rerank("测试查询")

        assert "未找到" in result

    def test_retrieve_and_rerank_none_results(self, rag_instance):
        """测试返回 None"""
        rag_instance.collection.query.return_value = None

        result = rag_instance.retrieve_and_rerank("测试查询")

        assert "未找到" in result

    def test_retrieve_and_rerank_default_top_k(self, rag_instance):
        """测试默认 top_k 值"""
        from config import Config

        rag_instance.collection.query.return_value = {
            'documents': [['doc1', 'doc2', 'doc3', 'doc4', 'doc5']]
        }
        rag_instance.ranker.rerank.return_value = [
            {'text': f'doc{i}', 'score': 0.9 - i * 0.1}
            for i in range(5)
        ]

        rag_instance.retrieve_and_rerank("测试")

        # 验证 rerank 被调用，结果会被切片到 top_k
        rag_instance.ranker.rerank.assert_called_once()

    def test_retrieve_and_rerank_failure(self, rag_instance):
        """测试检索失败时抛出异常"""
        from exceptions import RerankError

        rag_instance.collection.query.side_effect = Exception("检索失败")

        with pytest.raises(RerankError) as exc_info:
            rag_instance.retrieve_and_rerank("测试查询")

        assert "检索和重排序失败" in str(exc_info.value)

    def test_clear_db_success(self, rag_instance, mock_chromadb):
        """测试清空数据库成功"""
        rag_instance.clear_db()

        rag_instance.chroma_client.delete_collection.assert_called_once()
        rag_instance.chroma_client.create_collection.assert_called_once()

    def test_clear_db_failure(self, rag_instance):
        """测试清空数据库失败时抛出异常"""
        from exceptions import VectorStoreError

        rag_instance.chroma_client.delete_collection.side_effect = Exception("删除失败")

        with pytest.raises(VectorStoreError) as exc_info:
            rag_instance.clear_db()

        assert "清理向量数据库失败" in str(exc_info.value)
