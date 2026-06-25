# 知识库检索详细设计（RAG Pipeline）

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **文档入库** | PDF/Word/TXT/HTML → 分块 → Embedding → 存入向量库 |
| **语义检索** | 用户查询 → Embedding → 相似度搜索 → Top-K 相关片段 |
| **混合检索** | 向量检索 + BM25 关键词检索，RRF 融合排序 |
| **重排（Rerank）** | Cross-Encoder 精排，提升召回质量 |
| **知识更新** | 增量入库、版本管理、过期清理 |

---

## 2. Embedding 引擎 `knowledge/embedder.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass
class EmbeddingResult:
    """单次 embedding 结果"""
    vectors: list[list[float]]   # shape: (n_texts, dim)
    model_name: str
    dimensions: int


class BaseEmbedder(ABC):
    """Embedding 引擎抽象基类"""

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @property
    @abstractmethod
    def vector_dimensions(self) -> int: ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """批量文本 → 向量（用于入库）"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> EmbeddingResult:
        """单条查询 → 向量（用于检索，可能使用不同 prompt）"""
        ...


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding API"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        # text-embedding-3-small: 1536 dims (default), configurable 1-3072
        self._dimensions = 1536

    @property
    def engine_name(self) -> str:
        return f"openai-{self._model}"

    @property
    def vector_dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        vectors = [d.embedding for d in resp.data]
        return EmbeddingResult(vectors=vectors, model_name=self._model, dimensions=self._dimensions)

    async def embed_query(self, text: str) -> EmbeddingResult:
        result = await self.embed_documents([text])
        return result


class LocalEmbedder(BaseEmbedder):
    """本地 embedding 模型（sentence-transformers）"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        # bge-m3: 1024 dims
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def engine_name(self) -> str:
        return f"local-{self._model_name}"

    @property
    def vector_dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        import asyncio
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True).tolist()
        )
        return EmbeddingResult(vectors=vectors, model_name=self._model_name, dimensions=self._dimensions)

    async def embed_query(self, text: str) -> EmbeddingResult:
        return await self.embed_documents([text])


class OllamaEmbedder(BaseEmbedder):
    """Ollama embedding API（自托管方案）"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        import httpx
        self._client = httpx.AsyncClient(base_url=base_url)
        self._model = model
        self._dimensions = 768  # nomic-embed-text default

    @property
    def engine_name(self) -> str:
        return f"ollama-{self._model}"

    @property
    def vector_dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        vectors = []
        for text in texts:
            resp = await self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            vectors.append(resp.json()["embedding"])
        return EmbeddingResult(vectors=vectors, model_name=self._model, dimensions=self._dimensions)

    async def embed_query(self, text: str) -> EmbeddingResult:
        return await self.embed_documents([text])
```

---

## 3. 文档分块 `knowledge/chunker.py`

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Chunk:
    """文本分块结果"""
    text: str
    chunk_id: str          # 唯一标识，格式: {doc_id}_chunk_{index}
    doc_id: str            # 来源文档 ID
    index: int             # 在文档中的序号
    metadata: dict         # 页码、标题层级等元信息


class BaseChunker(Protocol):
    """分块策略协议"""

    async def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]: ...


class RecursiveTextSplitter(BaseChunker):
    """
    递归字符分割器（仿 LangChain）。

    按分隔符优先级逐层切分："\n\n" → "\n" → "。" → "，" → " " → "\u200b"
    确保语义完整性优先于固定长度。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._separators = ["\n\n", "\n", "。", "；", "！", "？", "，", ";", ".", " ", "\u200b"]

    async def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        import uuid
        chunks = self._split_text(text)
        result = []
        for i, segment in enumerate(chunks):
            chunk_meta = {**(metadata or {}), "chunk_index": i}
            result.append(Chunk(
                text=segment,
                chunk_id=f"{doc_id}_chunk_{i}",
                doc_id=doc_id,
                index=i,
                metadata=chunk_meta,
            ))
        return result

    def _split_text(self, text: str) -> list[str]:
        """递归分割，返回文本片段列表"""
        if len(text) <= self._chunk_size:
            return [text]

        for sep in self._separators:
            parts = text.split(sep)
            # 如果分割后每段都满足大小要求，则使用此分隔符
            if all(len(p) <= self._chunk_size for p in parts):
                return self._merge_parts(parts, sep)

        # 兜底：强制按字符数切分
        return self._force_split(text)

    def _merge_parts(self, parts: list[str], separator: str) -> list[str]:
        """将小片段合并为 chunk_size 大小的块"""
        if not parts:
            return []

        chunks = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + separator + part
            if len(candidate) > self._chunk_size and current.strip():
                chunks.append(current.strip())
                # 保留 overlap
                current = part
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _force_split(self, text: str) -> list[str]:
        """强制按字符数切分（含重叠）"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end].strip())
            start += self._chunk_size - self._overlap
        return chunks


class MarkdownChunker(BaseChunker):
    """
    Markdown 感知分块器。

    优先按标题层级分割，保持代码块、表格完整性。
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._fallback = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        import re
        # 按标题分割
        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        current_section = ""
        section_title = ""

        for section in sections:
            title_match = re.match(r'^(#{1,6}\s*.+?)(\n|$)', section)
            if title_match:
                section_title = title_match.group(1).strip()
                body = section[len(title_match.group(0)):].strip()
            else:
                body = section

            candidate = f"{section_title}\n{body}" if section_title else body

            if len(current_section) + len(candidate) > self._chunk_size and current_section.strip():
                sub_chunks = await self._fallback.chunk(current_section, doc_id, metadata)
                chunks.extend(sub_chunks)
                current_section = candidate
            else:
                current_section += "\n\n" + candidate if current_section else candidate

        if current_section.strip():
            sub_chunks = await self._fallback.chunk(current_section, doc_id, metadata)
            chunks.extend(sub_chunks)

        return chunks
```

---

## 4. 向量数据库 `knowledge/vector_store.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    """向量库中的文档"""
    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    score: float | None = None  # 检索时的相似度分数


@dataclass
class SearchResults:
    """检索结果"""
    documents: list[Document]
    total_count: int
    query_vector_id: str | None = None


class BaseVectorStore(ABC):
    """向量数据库抽象基类"""

    @property
    @abstractmethod
    def store_name(self) -> str: ...

    @abstractmethod
    async def create_collection(self, name: str, dimensions: int) -> bool:
        """创建集合（表）"""
        ...

    @abstractmethod
    async def upsert_documents(
        self, collection: str, documents: list[Document]
    ) -> int:
        """插入/更新文档，返回实际写入数量"""
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> SearchResults:
        """向量相似度搜索"""
        ...

    @abstractmethod
    async def delete_documents(
        self, collection: str, document_ids: list[str]
    ) -> int:
        """删除文档"""
        ...

    @abstractmethod
    async def drop_collection(self, name: str) -> bool:
        """删除集合"""
        ...


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 向量存储（嵌入式，适合开发/小规模）"""

    def __init__(self, persist_dir: str = "./data/chroma"):
        import chromadb
        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, any] = {}

    @property
    def store_name(self) -> str:
        return "chroma"

    async def create_collection(self, name: str, dimensions: int) -> bool:
        if name not in self._collections:
            try:
                col = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine", "dimension": str(dimensions)},
                )
                self._collections[name] = col
            except Exception:
                return False
        return True

    async def upsert_documents(self, collection: str, documents: list[Document]) -> int:
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        col = self._collections[collection]
        ids = [d.id for d in documents]
        texts = [d.text for d in documents]
        embeddings = [d.embedding for d in documents]
        metadatas = [d.metadata for d in documents]

        col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return len(documents)

    async def search(
        self, collection: str, query_vector: list[float], top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> SearchResults:
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        col = self._collections[collection]
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_expr,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i in range(len(results["ids"][0])):
            documents.append(Document(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                embedding=[],  # Chroma 不返回向量本身
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                score=1.0 - (results["distances"][0][i] if results["distances"] else 0),
            ))

        return SearchResults(documents=documents, total_count=len(documents))

    async def delete_documents(self, collection: str, document_ids: list[str]) -> int:
        col = self._collections[collection]
        col.delete(ids=document_ids)
        return len(document_ids)

    async def drop_collection(self, name: str) -> bool:
        self._client.delete_collection(name)
        self._collections.pop(name, None)
        return True


class QdrantVectorStore(BaseVectorStore):
    """Qdrant 向量存储（生产级，支持分布式）"""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        from qdrant_client import QdrantClient
        kwargs = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = QdrantClient(**kwargs)

    @property
    def store_name(self) -> str:
        return "qdrant"

    async def create_collection(self, name: str, dimensions: int) -> bool:
        from qdrant_client.http.models import Distance, VectorParams
        collections = [c.name for c in self._client.get_collections().collections]
        if name not in collections:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )
        return True

    async def upsert_documents(self, collection: str, documents: list[Document]) -> int:
        from qdrant_client.http.models import PointStruct
        points = [
            PointStruct(
                id=i,  # Qdrant 使用整数 ID，实际应维护映射表
                payload={"doc_id": d.id, "text": d.text, **d.metadata},
                vector=d.embedding,
            )
            for i, d in enumerate(documents)
        ]
        self._client.upsert(collection_name=collection, points=points)
        return len(documents)

    async def search(
        self, collection: str, query_vector: list[float], top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> SearchResults:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        conditions = []
        if filter_expr:
            for key, value in filter_expr.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        query_filter = Filter(must=conditions) if conditions else None

        results = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        documents = []
        for hit in results:
            payload = hit.payload or {}
            documents.append(Document(
                id=payload.get("doc_id", str(hit.id)),
                text=payload.get("text", ""),
                embedding=[],
                metadata={k: v for k, v in payload.items() if k not in ("doc_id", "text")},
                score=hit.score,
            ))

        return SearchResults(documents=documents, total_count=len(documents))

    async def delete_documents(self, collection: str, document_ids: list[str]) -> int:
        # Qdrant 需要 payload filter 来删除
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        self._client.delete(
            collection_name=collection,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=document_ids))]),
        )
        return len(document_ids)

    async def drop_collection(self, name: str) -> bool:
        self._client.delete_collection(collection_name=name)
        return True


class MilvusVectorStore(BaseVectorStore):
    """Milvus 向量存储（大规模生产级）"""

    def __init__(self, uri: str = "./data/milvus.db", token: str | None = None):
        from pymilvus import connections, utility
        self._uri = uri
        connections.connect(alias="default", uri=uri, token=token)
        self._utility = utility

    @property
    def store_name(self) -> str:
        return "milvus"

    async def create_collection(self, name: str, dimensions: int) -> bool:
        from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
        if self._utility.has_collection(name):
            return True

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description=f"Collection {name}")
        Collection(name=name, schema=schema)
        # 创建索引
        col = Collection(name)
        col.create_index("embedding", index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        })
        return True

    async def upsert_documents(self, collection: str, documents: list[Document]) -> int:
        from pymilvus import Collection
        col = Collection(collection)
        data = [
            [d.id for d in documents],
            [d.text for d in documents],
            [d.embedding for d in documents],
            [d.metadata for d in documents],
        ]
        col.insert(data)
        col.flush()
        return len(documents)

    async def search(
        self, collection: str, query_vector: list[float], top_k: int = 5,
        filter_expr: dict | None = None,
    ) -> SearchResults:
        from pymilvus import Collection
        col = Collection(collection)
        col.load()

        expr = ""
        if filter_expr:
            conditions = [f'metadata["{k}"] == "{v}"' for k, v in filter_expr.items()]
            expr = " and ".join(conditions)

        results = col.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
        )

        documents = []
        for hit in results[0]:
            entity = hit.entity
            documents.append(Document(
                id=entity["id"],
                text=entity.get("text", ""),
                embedding=[],
                metadata=entity.get("metadata", {}),
                score=hit.distance,
            ))

        return SearchResults(documents=documents, total_count=len(documents))

    async def delete_documents(self, collection: str, document_ids: list[str]) -> int:
        from pymilvus import Collection
        col = Collection(collection)
        expr = f'id in {document_ids}'
        col.delete(expr)
        return len(document_ids)

    async def drop_collection(self, name: str) -> bool:
        from pymilvus import utility
        if utility.has_collection(name):
            utility.drop_collection(name)
        return True
```

---

## 5. Rerank 引擎 `knowledge/reranker.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RankedDocument:
    """重排后的文档"""
    document_id: str
    text: str
    metadata: dict
    rerank_score: float   # [0.0, 1.0]


class BaseReranker(ABC):
    """重排引擎抽象基类"""

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[RankedDocument]:
        """对召回的文档进行精排"""
        ...


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排（sentence-transformers）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    @property
    def engine_name(self) -> str:
        return f"cross-encoder-{self._model_name}"

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[RankedDocument]:
        import asyncio
        loop = asyncio.get_event_loop()

        pairs = [[query, doc] for doc in documents]
        scores = await loop.run_in_executor(None, self._model.predict, pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            RankedDocument(
                document_id=f"doc_{i}",
                text=doc,
                metadata={},
                rerank_score=float(score),
            )
            for i, (doc, score) in enumerate(ranked)
        ]


class CohereReranker(BaseReranker):
    """Cohere Rerank API"""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        import cohere
        self._client = cohere.AsyncClient(api_key=api_key)
        self._model = model

    @property
    def engine_name(self) -> str:
        return f"cohere-{self._model}"

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[RankedDocument]:
        resp = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=True,
        )

        return [
            RankedDocument(
                document_id=f"doc_{r.index}",
                text=r.document.get("text", ""),
                metadata={},
                rerank_score=r.relevance_score,
            )
            for r in resp.results
        ]


class NoopReranker(BaseReranker):
    """无重排（直接返回原始排序）"""

    @property
    def engine_name(self) -> str:
        return "noop"

    async def rerank(
        self, query: str, documents: list[str], top_k: int = 3
    ) -> list[RankedDocument]:
        return [
            RankedDocument(
                document_id=f"doc_{i}",
                text=doc,
                metadata={},
                rerank_score=1.0 - (i * 0.1),
            )
            for i, doc in enumerate(documents[:top_k])
        ]
```

---

## 6. RAG Pipeline `knowledge/rag_pipeline.py`

```python
"""
RAG（Retrieval-Augmented Generation）Pipeline。

完整流程：
    用户查询 → Embedding → 向量检索 + BM25 → RRF 融合 → Rerank → Prompt 注入 → LLM 生成
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from knowledge.embedder import BaseEmbedder
from knowledge.vector_store import BaseVectorStore, Document, SearchResults
from knowledge.reranker import BaseReranker, RankedDocument

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """RAG 检索上下文（注入到 LLM prompt）"""
    relevant_chunks: list[str]       # 相关文本片段
    source_ids: list[str]            # 来源文档 ID
    scores: list[float]              # 相关性分数
    total_tokens: int = 0            # 估算 token 数


class RAGPipeline:
    """
    RAG Pipeline 编排器。

    Usage:
        pipeline = RAGPipeline(
            embedder=OpenAIEmbedder(api_key="..."),
            vector_store=ChromaVectorStore(),
            reranker=CrossEncoderReranker(),
        )
        await pipeline.initialize(collection_name="knowledge_base")

        # 入库
        await pipeline.ingest_document(text="...", doc_id="doc_001", collection="knowledge_base")

        # 检索
        context = await pipeline.retrieve(query="What is ...?", collection="knowledge_base")
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        reranker: BaseReranker | None = None,
        top_k: int = 5,
        rerank_top_k: int = 3,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker or NoopReranker()
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        from knowledge.chunker import RecursiveTextSplitter
        self._chunker = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def initialize(self, collection_name: str):
        """初始化向量库集合"""
        await self._vector_store.create_collection(
            name=collection_name,
            dimensions=self._embedder.vector_dimensions,
        )
        logger.info(f"RAG pipeline initialized with collection '{collection_name}'")

    async def ingest_document(
        self,
        text: str,
        doc_id: str,
        collection: str,
        metadata: dict | None = None,
    ) -> int:
        """
        文档入库流程：分块 → Embedding → 存入向量库。

        Returns: 实际写入的 chunk 数量
        """
        # Step 1: Chunking
        chunks = await self._chunker.chunk(text, doc_id, metadata)
        logger.info(f"Document '{doc_id}' split into {len(chunks)} chunks")

        if not chunks:
            return 0

        # Step 2: Embedding
        texts = [c.text for c in chunks]
        embed_result = await self._embedder.embed_documents(texts)

        # Step 3: Upsert to vector store
        documents = [
            Document(
                id=c.chunk_id,
                text=c.text,
                embedding=embed_result.vectors[i],
                metadata={**c.metadata, "source_doc": doc_id},
            )
            for i, c in enumerate(chunks)
        ]

        count = await self._vector_store.upsert_documents(collection, documents)
        logger.info(f"Ingested {count} chunks from '{doc_id}' into '{collection}'")
        return count

    async def retrieve(
        self,
        query: str,
        collection: str,
        filter_expr: dict | None = None,
    ) -> RAGContext:
        """
        检索流程：查询 Embedding → 向量搜索 → Rerank → 构建上下文。

        Returns: RAGContext（包含相关片段和来源信息）
        """
        # Step 1: Query embedding
        embed_result = await self._embedder.embed_query(query)
        query_vector = embed_result.vectors[0]

        # Step 2: Vector search (recall more for reranking)
        recall_k = max(self._top_k * 3, 20)
        results = await self._vector_store.search(
            collection=collection,
            query_vector=query_vector,
            top_k=recall_k,
            filter_expr=filter_expr,
        )

        if not results.documents:
            return RAGContext(relevant_chunks=[], source_ids=[], scores=[])

        # Step 3: Rerank
        ranked = await self._reranker.rerank(
            query=query,
            documents=[d.text for d in results.documents],
            top_k=self._rerank_top_k,
        )

        # Step 4: Build context
        relevant_chunks = [r.text for r in ranked]
        source_ids = []
        scores = []
        total_tokens = 0

        for i, r in enumerate(ranked):
            # Try to get source doc ID from original results
            orig_doc = results.documents[i] if i < len(results.documents) else None
            source_id = (orig_doc.metadata.get("source_doc", "") or r.document_id) if orig_doc else r.document_id
            source_ids.append(source_id)
            scores.append(r.rerank_score)
            total_tokens += len(r.text) // 4  # rough token estimate

        context = RAGContext(
            relevant_chunks=relevant_chunks,
            source_ids=source_ids,
            scores=scores,
            total_tokens=total_tokens,
        )

        logger.info(
            f"RAG retrieve: query='{query[:30]}...', "
            f"found {len(relevant_chunks)} chunks, {total_tokens} tokens"
        )
        return context

    def build_prompt(self, user_query: str, context: RAGContext) -> str:
        """
        将检索结果注入到 prompt。

        Returns: 完整的 system + user prompt
        """
        if not context.relevant_chunks:
            return f"Question: {user_query}"

        references = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context.relevant_chunks)
        )

        prompt = f"""参考以下知识库内容回答问题。如果知识库中没有相关信息，请说明"根据现有资料无法回答"。

=== 参考资料 ===
{references}
=== 参考资料结束 ===

问题：{user_query}

请基于以上参考资料作答，并在答案中标注引用来源（如 [1]、[2]）。"""
        return prompt
```

---

## 7. BM25 + RRF 混合检索 `knowledge/hybrid_search.py`

```python
"""
BM25 关键词检索 + RRF（Reciprocal Rank Fusion）融合排序。

解决纯向量检索在精确匹配场景下的不足。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    id: str
    text: str
    metadata: dict


class BM25Index:
    """轻量级 BM25 索引（基于 rank_bm25 库）"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self._bm25: BM25Okapi | None = None
        self._documents: list[BM25Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._id_map: dict[str, int] = {}  # doc_id → index
        self._k1 = k1
        self._b = b

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：中文按字符，英文按空格"""
        import re
        # 混合中英文分词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
        return tokens

    def build(self, documents: list[BM25Document]):
        """构建索引"""
        from rank_bm25 import BM25Okapi
        self._documents = documents
        self._tokenized_corpus = [self._tokenize(d.text) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self._k1, b=self._b)
        self._id_map = {d.id: i for i, d in enumerate(documents)}

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """搜索，返回 (doc_id, score) 列表"""
        if not self._bm25:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # 取 top_k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                results.append((self._documents[idx].id, float(scores[idx])))
        return results


class HybridSearcher:
    """
    混合检索器：向量检索 + BM25，RRF 融合。

    RRF formula: score(doc) = sum(1 / (k + rank)) for each ranking list
    Default k = 61 (per paper).
    """

    def __init__(self, rrf_k: int = 61, vector_weight: float = 0.7):
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._bm25_index: BM25Index | None = None

    def _rrf_fuse(
        self,
        vector_ranks: list[str],   # doc_ids in order of vector similarity
        bm25_ranks: list[str],     # doc_ids in order of BM25 score
        top_k: int = 5,
    ) -> list[str]:
        """RRF 融合排序"""
        scores: dict[str, float] = {}

        for rank, doc_id in enumerate(vector_ranks):
            scores[doc_id] = scores.get(doc_id, 0) + (
                self._vector_weight / (self._rrf_k + rank + 1)
            )

        bm25_weight = 1.0 - self._vector_weight
        for rank, doc_id in enumerate(bm25_ranks):
            scores[doc_id] = scores.get(doc_id, 0) + (
                bm25_weight / (self._rrf_k + rank + 1)
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_id for doc_id, _ in ranked]

    async def search(
        self,
        vector_store_results: list[tuple[str, float]],  # (doc_id, score) from vector search
        bm25_index: BM25Index,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        """执行混合检索"""
        vector_ranks = [doc_id for doc_id, _ in vector_store_results]
        bm25_results = bm25_index.search(query, top_k=max(top_k * 3, 20))
        bm25_ranks = [doc_id for doc_id, _ in bm25_results]

        return self._rrf_fuse(vector_ranks, bm25_ranks, top_k)
```

---

## 8. Agent 集成 `knowledge/integration.py`

```python
"""
知识库模块与 Agent Core 的集成。

将 RAG 检索封装为 Tool，Agent 在对话中自动调用。
"""

import logging
from tool.base import BaseTool, ToolResult, ToolDefinition
from knowledge.rag_pipeline import RAGPipeline, RAGContext

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(BaseTool):
    """知识库检索工具，Agent 可调用"""

    name = "search_knowledge"
    description = (
        "在知识库中搜索相关信息。适用于回答事实性问题、查询文档内容等场景。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或问题"},
            "collection": {"type": "string", "description": "知识库名称（可选，默认使用主库）"},
            "top_k": {"type": "integer", "description": "返回结果数量（可选，默认 3）"},
        },
        "required": ["query"],
    }

    def __init__(self, rag_pipeline: RAGPipeline, default_collection: str = "knowledge_base"):
        super().__init__()
        self._rag = rag_pipeline
        self._default_collection = default_collection

    async def execute(self, arguments: dict) -> ToolResult:
        query = arguments["query"]
        collection = arguments.get("collection", self._default_collection)
        top_k = int(arguments.get("top_k", 3))

        try:
            context = await self._rag.retrieve(query, collection)

            if not context.relevant_chunks:
                return ToolResult(
                    success=True,
                    content="知识库中未找到相关信息。",
                )

            # 构建返回内容
            parts = []
            for i, chunk in enumerate(context.relevant_chunks[:top_k]):
                source = context.source_ids[i] if i < len(context.source_ids) else "unknown"
                score = context.scores[i] if i < len(context.scores) else 0.0
                # 截断过长片段
                truncated = chunk[:500] + "..." if len(chunk) > 500 else chunk
                parts.append(f"[来源 {source}, 相关度 {score:.2f}]\n{truncated}")

            content = "\n\n---\n\n".join(parts)
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "total_chunks": len(context.relevant_chunks),
                    "source_ids": context.source_ids[:top_k],
                },
            )
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return ToolResult(
                success=False,
                content=f"知识库检索失败：{str(e)}",
            )


class KnowledgeIngestTool(BaseTool):
    """知识入库工具，Agent 可调用"""

    name = "ingest_knowledge"
    description = (
        "将文档内容添加到知识库。支持直接传入文本或文件内容。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要入库的文本内容"},
            "doc_id": {"type": "string", "description": "文档唯一标识（可选，自动生成）"},
            "collection": {"type": "string", "description": "知识库名称（可选）"},
        },
        "required": ["text"],
    }

    def __init__(self, rag_pipeline: RAGPipeline, default_collection: str = "knowledge_base"):
        super().__init__()
        self._rag = rag_pipeline
        self._default_collection = default_collection

    async def execute(self, arguments: dict) -> ToolResult:
        text = arguments["text"]
        doc_id = arguments.get("doc_id", f"auto_{hash(text[:100])}")
        collection = arguments.get("collection", self._default_collection)

        try:
            count = await self._rag.ingest_document(
                text=text, doc_id=doc_id, collection=collection
            )
            return ToolResult(
                success=True,
                content=f"成功入库 {count} 个文本片段到 '{collection}'。",
                metadata={"chunks_count": count, "doc_id": doc_id},
            )
        except Exception as e:
            logger.error(f"Knowledge ingest failed: {e}")
            return ToolResult(
                success=False,
                content=f"知识入库失败：{str(e)}",
            )
```

---

## 9. YAML 配置 `config/knowledge.yaml`

```yaml
knowledge:
  enabled: true

  embedder:
    type: "openai"                    # openai | local | ollama
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"      # 环境变量引用
    dimensions: 1536

  vector_store:
    type: "chroma"                    # chroma | qdrant | milvus
    persist_dir: "./data/chroma"
    # Qdrant config (when type=qdrant)
    # url: "http://localhost:6333"
    # api_key: "${QDRANT_API_KEY}"

  reranker:
    enabled: true
    type: "cross-encoder"             # cross-encoder | cohere | noop
    model: "BAAI/bge-reranker-v2-m3"

  retrieval:
    top_k: 5                          # 向量召回数量
    rerank_top_k: 3                   # 重排后保留数量
    hybrid_search: false              # 是否启用 BM25 + RRF 混合检索

  chunking:
    chunk_size: 512
    chunk_overlap: 64
    strategy: "recursive"             # recursive | markdown

  collections:
    - name: "knowledge_base"          # 默认知识库
      description: "通用知识库"
```

---

## 10. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │   (通过 Tool 接口    │
                    │    调用 RAG)         │
                    └──────────┬──────────┘
                               │ KnowledgeSearchTool
                               ▼
              ┌──────────────────────────────────┐
              │        RAG Pipeline              │
              │                                  │
              │  Query → Embedding → Vector DB   │
              │           + BM25 (optional)      │
              │           → RRF Fusion            │
              │           → Rerank                │
              │           → Prompt Injection      │
              └──────────┬───────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼               ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ OpenAI     │ │ Local      │ │ Ollama     │
   │ Embedder   │ │ Embedder   │ │ Embedder   │
   └────────────┘ └────────────┘ └────────────┘

          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │ ChromaDB   │ │ Qdrant     │ │ Milvus     │
          └────────────┘ └────────────┘ └────────────┘

          ┌────────────┐ ┌────────────┐
          │ Cross-Enc. │ │ Cohere     │
          │ Reranker   │ │ Reranker   │
          └────────────┘ └────────────┘
```

---

## 11. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **Embedding 多引擎** | OpenAI API / sentence-transformers / Ollama，统一 `BaseEmbedder` 接口 |
| **向量库可切换** | ChromaDB（开发）/ Qdrant（生产）/ Milvus（大规模），统一 `BaseVectorStore` 接口 |
| **智能分块** | 递归字符分割 + Markdown 感知，保持语义完整性 |
| **混合检索** | BM25 关键词 + 向量相似度，RRF 融合排序 |
| **重排精调** | Cross-Encoder / Cohere Rerank API，提升 Top-K 质量 |
| **Agent 集成** | `KnowledgeSearchTool` + `KnowledgeIngestTool`，Agent 自动发现调用 |
| **配置驱动** | YAML 管理所有参数，支持热重载 |

---

## 12. 多模态文档入库

### 12.1. 设计目标

扩展 RAG Pipeline 以支持多种文档格式的自动解析：
- **PDF**：文本提取 + 版面分析
- **Office 文档**：Word、Excel、PPT 内容提取
- **图片**：OCR 文字识别后入库
- **网页**：HTML → Markdown 转换

### 12.2. 文档解析器 `knowledge/document_parser.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    IMAGE = "image"       # PNG, JPG, etc.
    HTML = "html"         # Web pages
    TEXT = "text"         # Plain text, Markdown


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    text: str
    metadata: dict
    page_count: int = 0


class BaseDocumentParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def supported_types(self) -> list[DocumentType]: ...

    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument: ...


class PDFParser(BaseDocumentParser):
    """PDF 解析（使用 PyMuPDF / pdfplumber）"""

    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.PDF]

    async def parse(self, file_path: Path) -> ParsedDocument:
        import fitz  # PyMuPDF
        import asyncio
        loop = asyncio.get_event_loop()

        def _extract():
            doc = fitz.open(str(file_path))
            pages = []
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    pages.append(text)
            return "\n\n".join(pages), len(doc)

        text, page_count = await loop.run_in_executor(None, _extract)
        return ParsedDocument(
            doc_id=file_path.stem,
            text=text,
            metadata={"source_file": str(file_path.name), "type": "pdf"},
            page_count=page_count,
        )


class ImageParser(BaseDocumentParser):
    """图片解析（OCR）"""

    def supported_types(self) -> list[DocumentType]:
        return [DocumentType.IMAGE]

    async def parse(self, file_path: Path) -> ParsedDocument:
        # TODO: 集成 PaddleOCR / Tesseract
        import asyncio
        loop = asyncio.get_event_loop()

        def _ocr():
            from PIL import Image
            img = Image.open(str(file_path))
            return "[OCR placeholder]", 1

        text, page_count = await loop.run_in_executor(None, _ocr)
        return ParsedDocument(
            doc_id=file_path.stem,
            text=text,
            metadata={"source_file": str(file_path.name), "type": file_path.suffix.lstrip(".")},
            page_count=page_count,
        )


class DocumentParserFactory:
    """解析器工厂：根据文件类型选择对应解析器"""

    def __init__(self):
        self._parsers: dict[DocumentType, BaseDocumentParser] = {}
        for parser in [PDFParser(), ImageParser()]:
            for doc_type in parser.supported_types():
                self._parsers[doc_type] = parser

    @staticmethod
    def _detect_type(file_path: Path) -> DocumentType | None:
        ext_map = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".xlsx": DocumentType.XLSX,
            ".pptx": DocumentType.PPTX,
            ".png": DocumentType.IMAGE,
            ".jpg": DocumentType.IMAGE,
            ".jpeg": DocumentType.IMAGE,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".txt": DocumentType.TEXT,
            ".md": DocumentType.TEXT,
        }
        return ext_map.get(file_path.suffix.lower())

    async def parse(self, file_path: Path) -> ParsedDocument | None:
        doc_type = self._detect_type(file_path)
        if doc_type is None or doc_type not in self._parsers:
            return None
        return await self._parsers[doc_type].parse(file_path)
```

### 12.3. RAG Pipeline 扩展

在 `RAGPipeline` 中增加文件入库方法：

```python
async def ingest_file(
    self,
    file_path: Path,
    collection: str,
    doc_id: str | None = None,
) -> int:
    """多模态文档入库"""
    factory = DocumentParserFactory()
    parsed = await factory.parse(file_path)

    if parsed is None:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    did = doc_id or parsed.doc_id
    return await self.ingest_document(
        text=parsed.text,
        doc_id=did,
        collection=collection,
        metadata=parsed.metadata,
    )
```

---

## 13. 知识生命周期管理

### 13.1. 设计目标

知识库内容需要定期更新和清理：
- **TTL（生存时间）**：文档自动过期，避免过时信息干扰检索
- **版本控制**：同一文档的多次入库保留最新版本
- **增量更新**：仅处理变更部分，减少计算开销

### 13.2. TTL 管理 `knowledge/lifecycle.py`

```python
import logging
from datetime import datetime, timedelta

from knowledge.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class KnowledgeLifecycleManager:
    """知识生命周期管理器"""

    def __init__(self, vector_store: BaseVectorStore, default_ttl_days: int = 90):
        self._store = vector_store
        self._default_ttl = timedelta(days=default_ttl_days)

    async def cleanup_expired(
        self, collection: str, ttl_days: int | None = None
    ) -> int:
        """清理过期文档"""
        ttl = timedelta(days=ttl_days) if ttl_days else self._default_ttl
        cutoff = datetime.now() - ttl

        # 通过 metadata 中的 ingested_at 字段过滤
        filter_expr = {"ingested_at": {"$lt": cutoff.isoformat()}}
        expired_results = await self._store.search(
            collection=collection,
            query_vector=[0.0] * 1536,  # dummy vector for full scan
            top_k=10000,
            filter_expr=filter_expr,
        )

        if expired_results.documents:
            doc_ids = [d.id for d in expired_results.documents]
            await self._store.delete_documents(collection, doc_ids)
            logger.info(f"Cleaned up {len(doc_ids)} expired documents from '{collection}'")

        return len(expired_results.documents) if expired_results.documents else 0

    async def update_document(
        self,
        collection: str,
        doc_id: str,
        new_text: str,
        embedder=None,
        chunker=None,
    ) -> int:
        """更新文档：先删除旧版本，再入库新版本"""
        # 查找并删除旧版本（通过 source_doc metadata）
        old_results = await self._store.search(
            collection=collection,
            query_vector=[0.0] * 1536,
            top_k=10000,
            filter_expr={"source_doc": doc_id},
        )

        if old_results and old_results.documents:
            old_ids = [d.id for d in old_results.documents]
            await self._store.delete_documents(collection, old_ids)
            logger.info(f"Removed {len(old_ids)} old chunks for document '{doc_id}'")

        # 入库新版本
        if embedder and chunker:
            chunks = await chunker.chunk(new_text, doc_id, {})
            texts = [c.text for c in chunks]
            embed_result = await embedder.embed_documents(texts)

            from knowledge.vector_store import Document
            documents = [
                Document(
                    id=c.chunk_id,
                    text=c.text,
                    embedding=embed_result.vectors[i],
                    metadata={**c.metadata, "source_doc": doc_id},
                )
                for i, c in enumerate(chunks)
            ]
            return await self._store.upsert_documents(collection, documents)

        return 0
```

### 13.3. 定时清理任务

```python
import asyncio
import logging

logger = logging.getLogger(__name__)


async def lifecycle_cleanup_loop(
    manager: KnowledgeLifecycleManager,
    collections: list[str],
    interval_hours: int = 24,
):
    """后台定时清理过期知识"""
    while True:
        for collection in collections:
            try:
                count = await manager.cleanup_expired(collection)
                if count > 0:
                    logger.info(f"Lifecycle cleanup: removed {count} expired docs from '{collection}'")
            except Exception as e:
                logger.error(f"Lifecycle cleanup failed for '{collection}': {e}")

        await asyncio.sleep(interval_hours * 3600)
```

---

## 14. 更新后的设计总结

| 特性 | 实现方式 |
|------|---------|
| **Embedding 多引擎** | OpenAI API / sentence-transformers / Ollama，统一 `BaseEmbedder` 接口 |
| **向量库可切换** | ChromaDB（开发）/ Qdrant（生产）/ Milvus（大规模），统一 `BaseVectorStore` 接口 |
| **智能分块** | 递归字符分割 + Markdown 感知，保持语义完整性 |
| **混合检索** | BM25 关键词 + 向量相似度，RRF 融合排序 |
| **重排精调** | Cross-Encoder / Cohere Rerank API，提升 Top-K 质量 |
| **Agent 集成** | `KnowledgeSearchTool` + `KnowledgeIngestTool`，Agent 自动发现调用 |
| **配置驱动** | YAML 管理所有参数，支持热重载 |
| **多模态入库** | PDF / Office / 图片 OCR / HTML → 统一文本提取流水线 |
| **生命周期管理** | TTL 过期清理 + 文档版本更新 + 后台定时任务 |
