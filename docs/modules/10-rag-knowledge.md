# RAG 与知识库详细设计

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **检索增强** | 将外部知识注入 Agent 上下文，提升回答准确性 |
| **多源支持** | 文档、数据库、API、网页均可作为知识来源 |
| **混合检索** | 向量相似度 + BM25 关键词双路召回，重排序融合 |
| **增量更新** | 知识库支持增删改查，无需全量重建索引 |
| **可追溯性** | 每条回答附带引用来源，便于验证 |

---

## 2. 协议设计 `rag/protocol.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Document:
    """知识文档（最小粒度单元）"""
    content: str                                  # 文本内容
    source: str                                   # 来源标识，如文件路径、URL
    metadata: dict[str, Any] | None = None        # 元信息：作者、时间、标签等
    doc_id: str | None = None                     # 唯一 ID（自动生成）


@dataclass(frozen=True)
class Chunk:
    """文档切片（向量索引的最小单元）"""
    content: str                                  # 切片文本
    chunk_id: str                                 # 唯一 ID
    source_doc_id: str                            # 所属文档 ID
    metadata: dict[str, Any] | None = None        # 继承文档元信息 + 位置信息
    embedding: list[float] | None = None          # 向量表示（索引时填充）


@dataclass(frozen=True)
class SearchResult:
    """单条检索结果"""
    chunk: Chunk                                  # 匹配的切片
    score: float                                  # 相关性分数 [0, 1]，越高越相关
    rank_method: str                              # 排序方式："vector" | "bm25" | "rrf"


@dataclass(frozen=True)
class RetrievalResult:
    """检索结果集"""
    results: list[SearchResult]                   # 排序后的结果列表
    query: str                                    # 原始查询
    total_time_ms: float                          # 检索耗时（毫秒）
```

---

## 3. 文档加载器 `rag/loaders.py`

```python
from abc import ABC, abstractmethod
from multimodal.protocol import FileParseResult
from rag.protocol import Document


class BaseLoader(ABC):
    """文档加载器基类"""

    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        """
        从指定来源加载文档。

        Args:
            source: 来源标识（文件路径、URL、数据库查询等）

        Returns:
            Document 列表
        """
        ...


class FileLoader(BaseLoader):
    """
    本地文件加载器。

    支持：PDF、DOCX、XLSX、TXT、CSV、Markdown、HTML。
    复用 multimodal/file_parser 的解析能力。
    """

    def __init__(self, file_parser=None):
        self._parser = file_parser

    async def load(self, source: str) -> list[Document]:
        from multimodal.file_parser import UniversalFileParser
        parser = self._parser or UniversalFileParser()

        with open(source, "rb") as f:
            file_data = f.read()

        result = await parser.parse(file_data, filename=source)
        return [
            Document(
                content=result.text_content,
                source=source,
                metadata={
                    "mime_type": result.mime_type,
                    "pages": result.pages,
                    **(result.metadata or {}),
                },
            )
        ]


class WebLoader(BaseLoader):
    """
    网页加载器。

    抓取 URL 内容，提取正文文本。
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def load(self, source: str) -> list[Document]:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(source)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else source

        return [
            Document(
                content=text,
                source=source,
                metadata={"title": title, "url": source},
            )
        ]


class DatabaseLoader(BaseLoader):
    """
    数据库加载器。

    执行 SQL 查询，将结果集转换为文档。
    """

    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    async def load(self, source: str) -> list[Document]:
        """
        source 格式：sql:<SQL语句>

        Example: "sql:SELECT id, title, content FROM articles WHERE status='published'"
        """
        if not source.startswith("sql:"):
            raise ValueError(f"DatabaseLoader requires 'sql:' prefix. Got: {source[:20]}")

        query = source[4:]
        import asyncio
        loop = asyncio.get_event_loop()

        def _execute():
            import sqlite3
            conn = sqlite3.connect(self._conn_str)
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            return columns, rows

        columns, rows = await loop.run_in_executor(None, _execute)

        documents = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            # 将结构化数据转为自然语言描述
            content_parts = [f"{k}: {v}" for k, v in row_dict.items() if v is not None]
            documents.append(
                Document(
                    content="\n".join(content_parts),
                    source=f"db:{query}",
                    metadata=row_dict,
                )
            )

        return documents


class DirectoryLoader(BaseLoader):
    """
    目录加载器。

    递归扫描目录，按文件类型分发到对应 loader。
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".csv", ".md", ".html", ".htm"}

    def __init__(self, recursive: bool = True):
        self._recursive = recursive

    async def load(self, source: str) -> list[Document]:
        import os
        from pathlib import Path

        loader = FileLoader()
        documents = []
        base = Path(source)

        pattern = "**/*" if self._recursive else "*"
        for file_path in base.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = await loader.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load {file_path}: {e}")

        return documents
```

---

## 4. 文档切片器 `rag/chunkers.py`

```python
from abc import ABC, abstractmethod
import uuid
from rag.protocol import Document, Chunk


class BaseChunker(ABC):
    """文档切片器基类"""

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        """将文档切分为多个 Chunk"""
        ...


class RecursiveTextChunker(BaseChunker):
    """
    递归文本切片器（langchain-style）。

    按分隔符层级递归分割，直到每个块小于 max_chunk_size。
    分隔符优先级：\n\n → \n → 。 → ， → 空格
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]

    async def chunk(self, document: Document) -> list[Chunk]:
        chunks = self._split_text(document.content, document.source or "")
        return chunks

    def _split_text(self, text: str, source_doc_id: str) -> list[Chunk]:
        """递归分割文本"""
        final_chunks = []
        self._recursive_split(text, self._separators, source_doc_id, final_chunks)
        return final_chunks

    def _recursive_split(
        self, text: str, separators: list[str], source_doc_id: str, output: list[Chunk]
    ):
        if len(text) <= self._chunk_size:
            if text.strip():
                output.append(self._make_chunk(text, source_doc_id))
            return

        for sep in separators:
            splits = text.split(sep)
            if len(splits) > 1:
                # 合并小块，保持重叠
                current = ""
                for split in splits:
                    if len(current) + len(split) + len(sep) <= self._chunk_size:
                        current += sep + split if current else split
                    else:
                        if current.strip():
                            output.append(self._make_chunk(current, source_doc_id))
                        # 保留重叠部分
                        overlap_start = max(0, self._chunk_size - self._overlap)
                        remaining = current[overlap_start:] if overlap_start < len(current) else split
                        current = remaining + sep + split if remaining else split

                if current.strip():
                    output.append(self._make_chunk(current, source_doc_id))
                return

        # 所有分隔符都无法分割，强制截断
        for i in range(0, len(text), self._chunk_size - self._overlap):
            block = text[i:i + self._chunk_size]
            if block.strip():
                output.append(self._make_chunk(block, source_doc_id))

    def _make_chunk(self, content: str, source_doc_id: str) -> Chunk:
        return Chunk(
            content=content.strip(),
            chunk_id=str(uuid.uuid4()),
            source_doc_id=source_doc_id,
        )


class SemanticChunker(BaseChunker):
    """
    语义切片器。

    基于句子边界 + 嵌入相似度进行智能分割，保持语义完整性。
    """

    def __init__(
        self,
        embedding_model,                              # EmbeddingModel 实例
        chunk_size: int = 500,                        # 每个 chunk 最大句子数
        similarity_threshold: float = 0.7,            # 相似度阈值，低于此值则新起一个 chunk
    ):
        self._embedder = embedding_model
        self._max_sentences = chunk_size
        self._threshold = similarity_threshold

    async def chunk(self, document: Document) -> list[Chunk]:
        import re

        # 按句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s*', document.content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks_sentences: list[list[str]] = [[sentences[0]]]

        for i in range(1, len(sentences)):
            prev_chunk = chunks_sentences[-1]

            # 如果当前 chunk 已满，开新 chunk
            if len(prev_chunk) >= self._max_sentences:
                chunks_sentences.append([sentences[i]])
                continue

            # 计算当前句子与前一个句子的相似度
            last_sentence = prev_chunk[-1]
            similarity = await self._sentence_similarity(last_sentence, sentences[i])

            if similarity >= self._threshold:
                prev_chunk.append(sentences[i])
            else:
                chunks_sentences.append([sentences[i]])

        return [
            Chunk(
                content=" ".join(chunk_sents),
                chunk_id=str(uuid.uuid4()),
                source_doc_id=document.source or "",
            )
            for chunk_sents in chunks_sentences
        ]

    async def _sentence_similarity(self, s1: str, s2: str) -> float:
        """计算两个句子的余弦相似度"""
        v1 = await self._embedder.embed(s1)
        v2 = await self._embedder.embed(s2)
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
```

---

## 5. Embedding 模型 `rag/embeddings.py`

```python
from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """嵌入模型基类"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """单文本 → 向量"""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表（默认逐条调用，可覆盖为真正的 batch）"""
        return [await self.embed(t) for t in texts]


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI Embedding API"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        # text-embedding-3-small: 1536 dim (default), text-embedding-3-large: 3072

    @property
    def dimension(self) -> int:
        if "large" in self._model:
            return 3072
        return 1536

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in resp.data]


class OllamaEmbedding(BaseEmbeddingModel):
    """Ollama Embedding（本地部署）"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    @property
    def dimension(self) -> int:
        # nomic-embed-text: 768 dim
        return 768

    async def embed(self, text: str) -> list[float]:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    本地 Sentence-Transformers 嵌入。

    适合离线场景，无需 API Key。
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self._model.encode(text).tolist()
        )
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts).tolist()
        )
        return [list(e) for e in embeddings]
```

---

## 6. 向量存储 `rag/vector_store.py`

```python
from abc import ABC, abstractmethod
import uuid
from rag.protocol import Chunk


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        """添加切片（返回 chunk_id 列表）"""
        ...

    @abstractmethod
    async def similarity_search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """向量相似度检索，返回 (chunk, score) 列表"""
        ...

    @abstractmethod
    async def delete_by_source(self, source_doc_id: str) -> int:
        """按来源文档 ID 删除切片，返回删除数量"""
        ...

    @abstractmethod
    async def count(self) -> int:
        """返回存储的切片总数"""
        ...


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB 向量存储。

    轻量级、嵌入式、无需外部服务。
    支持持久化到本地目录。
    """

    def __init__(self, collection_name: str = "agent_knowledge", persist_dir: str | None = None):
        import chromadb
        client_args = {}
        if persist_dir:
            client_args["path"] = persist_dir
        self._client = chromadb.Client() if not persist_dir else chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

    async def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding is not None]

        # 如果 chunk 没有 embedding，需要外部传入或报错
        metadatas = []
        for c in chunks:
            meta = dict(c.metadata or {})
            meta["source_doc_id"] = c.source_doc_id
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    async def similarity_search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            # Chroma 返回 distance，转换为相似度 score [0, 1]
            distance = results["distances"][0][i]
            score = 1.0 - distance  # cosine distance → similarity

            meta = results["metadatas"][0][i] if results["metadatas"][0][i] else {}
            chunk = Chunk(
                content=results["documents"][0][i],
                chunk_id=results["ids"][0][i],
                source_doc_id=meta.get("source_doc_id", ""),
                metadata={k: v for k, v in meta.items() if k != "source_doc_id"},
            )
            output.append((chunk, score))

        return output

    async def delete_by_source(self, source_doc_id: str) -> int:
        # Chroma 不支持直接按 metadata 删除，需要 workaround
        results = self._collection.get(where={"source_doc_id": source_doc_id})
        count = len(results["ids"])
        if count > 0:
            self._collection.delete(ids=results["ids"])
        return count

    async def count(self) -> int:
        return self._collection.count()


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant 向量存储。

    适合生产环境，支持分布式部署、过滤检索。
    """

    def __init__(
        self,
        collection_name: str = "agent_knowledge",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dimension: int = 1536,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self._client = QdrantClient(url=url, api_key=api_key)

        # 创建或获取 collection
        collections = self._client.get_collections()
        existing = {c.name for c in collections.collections}

        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

        self._collection_name = collection_name

    async def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        from qdrant_client.http.models import PointStruct

        if not chunks:
            return []

        points = [
            PointStruct(
                id=c.chunk_id,
                vector=c.embedding or [],
                payload={
                    "content": c.content,
                    "source_doc_id": c.source_doc_id,
                    **(c.metadata or {}),
                },
            )
            for c in chunks
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
        return [c.chunk_id for c in chunks]

    async def similarity_search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        output = []
        for hit in results:
            payload = hit.payload or {}
            chunk = Chunk(
                content=payload.get("content", ""),
                chunk_id=str(hit.id),
                source_doc_id=payload.get("source_doc_id", ""),
                metadata={k: v for k, v in payload.items() if k not in ("content", "source_doc_id")},
            )
            output.append((chunk, hit.score))

        return output

    async def delete_by_source(self, source_doc_id: str) -> int:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="source_doc_id", match=MatchValue(value=source_doc_id))
            ]),
        )
        return 0  # Qdrant delete 不返回删除数量

    async def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0


class SQLiteVectorStore(BaseVectorStore):
    """
    SQLite + sqlite-vec 向量存储。

    最轻量级方案，零依赖外部服务，适合开发/测试环境。
    """

    def __init__(self, db_path: str = ":memory:", dimension: int = 1536):
        import sqlite3
        self._conn = sqlite3.connect(db_path)
        self._dimension = dimension
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT,
                source_doc_id TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        self._conn.commit()

    async def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        import json
        for c in chunks:
            emb_blob = bytes().join(b.fromhex(format(b, '02x')) for b in c.embedding) if c.embedding else None
            # 简化：直接存储为 JSON text
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (c.chunk_id, c.content, c.source_doc_id, json.dumps(c.metadata or {}), json.dumps(c.embedding)),
            )
        self._conn.commit()
        return [c.chunk_id for c in chunks]

    async def similarity_search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        import json
        rows = self._conn.execute("SELECT chunk_id, content, source_doc_id, metadata, embedding FROM chunks").fetchall()

        scores = []
        qv = query_vector
        for row in rows:
            emb = json.loads(row[4])
            if not emb:
                continue
            dot = sum(a * b for a, b in zip(qv, emb))
            norm_q = sum(a * a for a in qv) ** 0.5
            norm_e = sum(b * b for b in emb) ** 0.5
            score = dot / (norm_q * norm_e) if norm_q > 0 and norm_e > 0 else 0

            meta = json.loads(row[3]) if row[3] else {}
            chunk = Chunk(
                content=row[1],
                chunk_id=row[0],
                source_doc_id=row[2],
                metadata=meta,
            )
            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def delete_by_source(self, source_doc_id: str) -> int:
        cursor = self._conn.execute("DELETE FROM chunks WHERE source_doc_id = ?", (source_doc_id,))
        self._conn.commit()
        return cursor.rowcount

    async def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0
```

---

## 7. BM25 检索 `rag/bm25.py`

```python
import math
import re
from collections import Counter


class BM25Retriever:
    """
    BM25 关键词检索。

    与向量检索互补：向量擅长语义匹配，BM25 擅长精确关键词匹配。
    两者结合通过 RRF（Reciprocal Rank Fusion）融合。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1  # 词频饱和参数
        self._b = b    # 长度归一化参数
        self._corpus: list[Chunk] = []
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0

    def index(self, chunks: list[Chunk]):
        """构建 BM25 索引"""
        from rag.protocol import Chunk

        self._corpus = list(chunks)
        num_docs = len(chunks)
        if num_docs == 0:
            return

        # 计算平均文档长度
        doc_lengths = [len(self._tokenize(c.content)) for c in chunks]
        self._avg_dl = sum(doc_lengths) / num_docs

        # 计算 IDF
        df = Counter()  # term → document frequency
        for chunk in chunks:
            tokens = set(self._tokenize(chunk.content))
            for token in tokens:
                df[token] += 1

        self._idf = {
            term: math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            for term, doc_freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """BM25 检索"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[Chunk, float]] = []

        for chunk in self._corpus:
            doc_tokens = self._tokenize(chunk.content)
            doc_len = len(doc_tokens)
            token_counts = Counter(doc_tokens)

            score = 0.0
            for q_token in query_tokens:
                tf = token_counts.get(q_token, 0)
                if tf == 0:
                    continue

                # BM25 公式
                idf = self._idf.get(q_token, 0.0)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((chunk, score))

        # 归一化分数到 [0, 1]
        max_score = max(s for _, s in scores) if scores else 1.0
        normalized = [(c, s / max_score) for c, s in scores]
        normalized.sort(key=lambda x: x[1], reverse=True)

        return normalized[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：中文按字符，英文按单词"""
        # 提取中英文混合 token
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        return tokens
```

---

## 8. RAG Pipeline `rag/pipeline.py`

```python
"""
RAG Pipeline：完整的检索增强生成流程。

Flow:
    Query → [向量检索] ─┐
            [BM25 检索] ─┼→ 重排序 (RRF) → Top-K Chunks → Prompt 注入 → LLM 回答
"""

import asyncio
import logging
import time
from rag.protocol import Chunk, SearchResult, RetrievalResult, Document
from rag.vector_store import BaseVectorStore
from rag.embeddings import BaseEmbeddingModel
from rag.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    混合检索 RAG Pipeline。

    Features:
    - 双路召回：向量相似度 + BM25 关键词
    - RRF 重排序融合
    - 引用来源追踪
    - 上下文窗口自适应截断
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        bm25_retriever: BM25Retriever | None = None,
        top_k: int = 5,
        max_context_tokens: int = 4000,
    ):
        self._vector_store = vector_store
        self._embedder = embedding_model
        self._bm25 = bm25_retriever
        self._top_k = top_k
        self._max_context_tokens = max_context_tokens

    async def retrieve(self, query: str) -> RetrievalResult:
        """
        执行混合检索。

        Args:
            query: 用户查询文本

        Returns:
            RetrievalResult（排序后的结果列表）
        """
        start_time = time.monotonic()

        # Step 1: 双路召回
        vector_results = await self._vector_search(query)

        bm25_results = []
        if self._bm25:
            bm25_results = self._bm25.search(query, top_k=self._top_k * 2)

        # Step 2: RRF 融合重排序
        fused = self._rrf_fuse(vector_results, bm25_results)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        search_results = [
            SearchResult(
                chunk=chunk,
                score=score,
                rank_method="rrf",
            )
            for chunk, score in fused[:self._top_k]
        ]

        return RetrievalResult(
            results=search_results,
            query=query,
            total_time_ms=elapsed_ms,
        )

    async def retrieve_and_format(self, query: str) -> tuple[str, list[dict]]:
        """
        检索并格式化为 Prompt 注入文本。

        Returns:
            (formatted_context, citations)
            - formatted_context: 可直接插入 system prompt 的上下文文本
            - citations: 引用来源列表 [{"source": "...", "content_snippet": "..."}]
        """
        result = await self.retrieve(query)

        if not result.results:
            return "", []

        # 按 token 预算截断
        chunks_to_use = self._fit_to_budget(result.results)

        context_parts = []
        citations = []

        for i, sr in enumerate(chunks_to_use, 1):
            snippet = sr.chunk.content[:500]  # 每条截取前 500 字符
            source = sr.chunk.source_doc_id or "unknown"
            context_parts.append(f"[{i}] (来源: {source}, 相关度: {sr.score:.2f})\n{snippet}")

            citations.append({
                "source": source,
                "content_snippet": snippet[:200],
                "score": sr.score,
            })

        formatted = "\n\n---\n\n".join(context_parts)
        return formatted, citations

    async def index_documents(self, documents: list[Document], chunker) -> int:
        """
        将文档切片并索引到向量存储。

        Args:
            documents: 原始文档列表
            chunker:   Chunker 实例

        Returns:
            索引的 chunk 总数
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            chunks = await chunker.chunk(doc)
            # 继承文档元信息
            for c in chunks:
                merged_meta = dict(doc.metadata or {})
                if c.metadata:
                    merged_meta.update(c.metadata)
                c = c._replace(metadata=merged_meta if merged_meta else None)

            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # 批量计算 embedding
        logger.info(f"Computing embeddings for {len(all_chunks)} chunks...")
        embeddings = await self._embedder.embed_batch([c.content for c in all_chunks])

        # 附加 embedding 到 chunk
        indexed_chunks = [
            c._replace(embedding=emb) for c, emb in zip(all_chunks, embeddings)
        ]

        # 写入向量存储
        await self._vector_store.add_chunks(indexed_chunks)

        # 更新 BM25 索引
        if self._bm25:
            self._bm25.index(indexed_chunks)

        logger.info(f"Indexed {len(indexed_chunks)} chunks from {len(documents)} documents")
        return len(indexed_chunks)

    async def delete_source(self, source_doc_id: str) -> int:
        """删除指定来源的所有切片"""
        count = await self._vector_store.delete_by_source(source_doc_id)
        if self._bm25:
            # 重建 BM25 索引（简单方案）
            remaining = await self._vector_store.similarity_search([0.0] * self._embedder.dimension, top_k=10000)
            self._bm25.index([c for c, _ in remaining])
        return count

    # ---------- 内部方法 ----------

    async def _vector_search(self, query: str) -> list[tuple[Chunk, float]]:
        """向量检索"""
        query_vector = await self._embedder.embed(query)
        results = await self._vector_store.similarity_search(
            query_vector, top_k=self._top_k * 2
        )
        return results

    @staticmethod
    def _rrf_fuse(
        vector_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion (RRF) 重排序。

        score = Σ 1 / (k + rank)
        """
        scores: dict[str, float] = {}  # chunk_id → rrf_score
        chunks_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(vector_results):
            cid = chunk.chunk_id
            chunks_map[cid] = chunk
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = chunk.chunk_id
            chunks_map[cid] = chunk
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)

        fused = [
            (chunks_map[cid], score)
            for cid, score in scores.items()
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def _fit_to_budget(self, results: list[SearchResult]) -> list[SearchResult]:
        """按 token 预算截断结果"""
        total_tokens = 0
        selected = []
        for sr in results:
            tokens = len(sr.chunk.content) // 4  # 粗略估算
            if total_tokens + tokens > self._max_context_tokens:
                break
            selected.append(sr)
            total_tokens += tokens
        return selected


# ==================== Prompt 注入模板 ====================

RAG_SYSTEM_PROMPT_TEMPLATE = """你是一个智能助手。在回答用户问题时，请优先参考以下检索到的相关知识。

## 参考资料
{context}

## 引用规则
- 如果参考资料中有相关信息，请在回答末尾标注引用来源，格式：[1]、[2] 等
- 如果参考资料不足以回答问题，可以基于你的知识补充，但需说明"以下内容来自我的训练数据而非参考资料"
- 不要编造不存在的资料

## 用户问题
{query}
"""


def build_rag_prompt(query: str, context: str) -> str:
    """构建 RAG 增强的系统提示词"""
    if not context.strip():
        return query  # 没有检索结果，直接返回原始查询

    return RAG_SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)
```

---

## 9. Agent 集成 `rag/integration.py`

```python
"""
RAG 模块与 Agent Core 的集成。

注册为 Tool：Agent 可在对话中主动调用知识检索。
同时支持自动 RAG：在每次 LLM 调用前自动注入相关知识。
"""

import logging
from tool.base import BaseTool, ToolResult, ToolDefinition
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(BaseTool):
    """知识检索工具，Agent 可主动调用"""

    name = "search_knowledge"
    description = "在知识库中搜索相关信息。适用于需要事实性知识的场景。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询文本"},
            "top_k": {"type": "integer", "description": "返回结果数量，默认 5", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, rag_pipeline: RAGPipeline):
        super().__init__()
        self._rag = rag_pipeline

    async def execute(self, arguments: dict) -> ToolResult:
        query = arguments["query"]
        top_k = int(arguments.get("top_k", 5))

        result = await self._rag.retrieve(query)

        if not result.results:
            return ToolResult(success=True, content="未找到相关知识。")

        parts = []
        for i, sr in enumerate(result.results[:top_k], 1):
            snippet = sr.chunk.content[:300]
            source = sr.chunk.source_doc_id or "unknown"
            parts.append(f"[{i}] (来源: {source})\n{snippet}")

        return ToolResult(
            success=True,
            content="\n\n".join(parts),
            metadata={
                "total_results": len(result.results),
                "query_time_ms": result.total_time_ms,
            },
        )


class AutoRAGMiddleware:
    """
    自动 RAG 中间件。

    在 Agent Core 的 LLM 调用链中拦截，自动执行检索并注入上下文。

    Usage (in agent_core):
        self._rag_middleware = AutoRAGMiddleware(rag_pipeline)
        # Before calling LLM:
        enhanced_messages, citations = await self._rag_middleware.enrich(messages)
    """

    def __init__(self, rag_pipeline: RAGPipeline, enabled: bool = True):
        self._rag = rag_pipeline
        self._enabled = enabled

    async def enrich(
        self, messages: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        增强消息列表，注入检索到的知识。

        Args:
            messages: Agent Core 的消息列表

        Returns:
            (enhanced_messages, citations)
        """
        if not self._enabled:
            return messages, []

        # 提取用户最后一条消息作为查询
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query.strip():
            return messages, []

        context, citations = await self._rag.retrieve_and_format(user_query)

        if not context:
            return messages, []

        # 在 system message 中注入知识（或创建新的 system message）
        enhanced = list(messages)
        rag_prompt = build_rag_prompt(user_query, context)

        # 查找或创建 system message
        system_found = False
        for i, msg in enumerate(enhanced):
            if msg.get("role") == "system":
                original_system = msg.get("content", "")
                enhanced[i] = {
                    **msg,
                    "content": f"{original_system}\n\n{rag_prompt}" if original_system else rag_prompt,
                }
                system_found = True
                break

        if not system_found:
            enhanced.insert(0, {"role": "system", "content": rag_prompt})

        return enhanced, citations

    def build_rag_prompt(self, query: str, context: str) -> str:
        """构建 RAG prompt（公开方法，供外部调用）"""
        from rag.pipeline import build_rag_prompt
        return build_rag_prompt(query, context)
```

---

## 10. 架构总览

```
                    ┌─────────────────────┐
                    │     Agent Core      │
                    │                     │
                    │  AutoRAGMiddleware  │◄── 自动注入知识上下文
                    │  KnowledgeSearchTool│◄── Agent 主动检索
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    RAG Pipeline     │
                    │                     │
                    │  Query → Embedding  │
                    │         ↓           │
                    │  ┌─────┴─────┐      │
                    │  │Vector Store│      │
                    │  │(Chroma/   │      │
                    │  │ Qdrant)   │      │
                    │  └─────┬─────┘      │
                    │         ↓           │
                    │  ┌─────┴─────┐      │
                    │  │  BM25     │      │
                    │  │Retriever  │      │
                    │  └─────┬─────┘      │
                    │         ↓           │
                    │  RRF Fusion → Top-K │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ FileLoader  │  │ WebLoader   │  │ DB Loader   │
     │ DirLoader   │  │             │  │             │
     └─────────────┘  └─────────────┘  └─────────────┘
              │                │                 │
              ▼                ▼                 ▼
     ┌──────────────────────────────────────────┐
     │           Chunker + Embedder             │
     │  RecursiveTextChunker / SemanticChunker  │
     │  OpenAIEmbedding / OllamaEmbedding       │
     └──────────────────────────────────────────┘
```

---

## 11. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **双路召回** | 向量相似度 + BM25 关键词，互补覆盖语义和精确匹配 |
| **RRF 融合** | Reciprocal Rank Fusion 重排序，无需训练参数 |
| **多源加载** | FileLoader / WebLoader / DB Loader / DirectoryLoader |
| **智能切片** | RecursiveTextChunker（规则）+ SemanticChunker（语义） |
| **多存储后端** | ChromaDB（轻量）/ Qdrant（生产）/ SQLite（零依赖） |
| **Agent 集成** | Tool 模式（主动检索）+ Middleware 模式（自动注入） |
| **引用追踪** | 每条结果附带来源信息，回答可追溯 |
| **增量更新** | 支持按文档 ID 增删改查索引 |
