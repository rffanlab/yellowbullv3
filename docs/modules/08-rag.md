# RAG（检索增强生成）详细设计

## 1. 职责边界

| 职责 | 说明 |
|------|------|
| **文档解析** | PDF、Word、Markdown、HTML 等格式文本提取 |
| **分块策略** | 按语义/固定大小/重叠窗口切分文档为 chunks |
| **Embedding** | 文本向量化，支持多 Embedding provider |
| **向量存储** | 本地 ChromaDB / 远程 Milvus、Pinecone 等 |
| **相似度检索** | cosine similarity / HNSW 近似最近邻搜索 |
| **Prompt 注入** | 检索结果拼接到 system prompt，供 LLM 引用 |

---

## 2. Embedding Provider ABC `rag/embedding.py`

```python
"""
Embedding provider abstraction。

支持：OpenAI text-embedding-3-small、本地 sentence-transformers、Azure OpenAI。
与 llm/provider_factory.py 模式一致，统一接口切换。
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """Embedding provider interface"""

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化（用于索引构建）"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """单条查询向量化（用于检索）"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = await self._client.embeddings.create(
            model=self._model, input=texts
        )
        return [e.embedding for e in resp.data]

    async def embed_query(self, text: str) -> List[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        # text-embedding-3-small: 1536, text-embedding-3-large: 3072
        return 1536 if "small" in self._model else 3072


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地 sentence-transformers provider（零 API cost）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def create_embedding_provider(config: dict) -> EmbeddingProvider:
    """Factory function"""
    provider_type = config.get("provider", "openai")

    if provider_type == "openai":
        return OpenAIEmbeddingProvider(
            api_key=config["api_key"],
            model=config.get("model", "text-embedding-3-small"),
        )
    elif provider_type == "local":
        return LocalEmbeddingProvider(
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")
```

---

## 3. 文档解析与分块 `rag/chunker.py`

```python
"""
Document parser and chunker。

支持格式：PDF (PyMuPDF), Markdown, HTML, TXT, DOCX (python-docx)
分块策略：固定大小 + 重叠窗口 / 语义边界（段落/标题）
"""

import re
from typing import List, Protocol


class DocumentChunk:
    """单个文本块"""
    def __init__(self, text: str, source: str, chunk_index: int, metadata: dict = None):
        self.text = text
        self.source = source          # 文件路径或 URL
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


class Chunker(Protocol):
    def chunk(self, text: str, source: str) -> List[DocumentChunk]: ...


class SlidingWindowChunker:
    """固定大小 + 重叠窗口分块"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        chunks = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 尽量在句子边界截断
            if end < len(text):
                last_period = text.rfind(".", start + 100, end)
                if last_period > start + 100:
                    end = last_period + 1
            chunks.append(DocumentChunk(
                text=text[start:end].strip(),
                source=source,
                chunk_index=index,
            ))
            start = end - self.overlap
            index += 1
        return chunks


class SemanticChunker:
    """按语义边界（标题、段落）分块"""

    def __init__(self, max_chunk_size: int = 1024):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        # Markdown: 按 ## 标题分割；纯文本: 按空行分割
        if "#" in text:
            sections = re.split(r"(?=#{1,6}\s)", text)
        else:
            sections = text.split("\n\n")

        chunks = []
        index = 0
        current_text = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(current_text) + len(section) > self.max_chunk_size:
                if current_text:
                    chunks.append(DocumentChunk(
                        text=current_text.strip(), source=source, chunk_index=index
                    ))
                    index += 1
                current_text = section
            else:
                current_text += "\n\n" + section

        if current_text:
            chunks.append(DocumentChunk(
                text=current_text.strip(), source=source, chunk_index=index
            ))

        return chunks


def parse_document(file_path: str) -> str:
    """解析文档为纯文本"""
    ext = file_path.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == "md" or ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == "html" or ext == "htm":
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n")
    elif ext == "docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported document format: {ext}")
```

---

## 4. 向量存储 `rag/vector_store.py`

```python
"""
Vector store abstraction。

支持：ChromaDB（本地）、Milvus（远程）、内存存储（测试/开发）。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class VectorStore(ABC):
    """向量存储接口"""

    @abstractmethod
    async def add_documents(self, chunks: List["DocumentChunk"], collection: str = "default"):
        """添加文档块（自动 embedding + 存储）"""
        ...

    @abstractmethod
    async def search(
        self, query: str, collection: str = "default",
        top_k: int = 5, score_threshold: float = 0.7,
    ) -> List[dict]:
        """相似度检索，返回 [(text, score, metadata), ...]"""
        ...

    @abstractmethod
    async def delete_collection(self, collection: str):
        """删除整个集合"""
        ...

    @abstractmethod
    async def count(self, collection: str = "default") -> int:
        """返回集合中文档块数量"""
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB 本地向量存储"""

    def __init__(self, embedding_provider: "EmbeddingProvider", persist_dir: str = "./data/chroma"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = embedding_provider

    async def add_documents(self, chunks: List["DocumentChunk"], collection: str = "default"):
        col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [c.text for c in chunks]
        embeddings = await self._embedder.embed_documents(texts)
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]
        ids = [f"{collection}_{i}" for i in range(len(chunks))]

        col.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

    async def search(self, query: str, collection: str = "default", top_k: int = 5, score_threshold: float = 0.7):
        col = self._client.get_collection(name=collection)
        query_embedding = await self._embedder.embed_query(query)

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        # ChromaDB cosine distance: 0=相同, 2=相反 → 转换为 similarity score
        return [
            {
                "text": doc,
                "score": round(1 - dist / 2, 4),
                "metadata": meta,
            }
            for doc, dist, meta in zip(
                results["documents"][0], results["distances"][0], results["metadatas"][0]
            )
            if (1 - dist / 2) >= score_threshold
        ]

    async def delete_collection(self, collection: str):
        self._client.delete_collection(name=collection)

    async def count(self, collection: str = "default") -> int:
        try:
            col = self._client.get_collection(name=collection)
            return col.count()
        except Exception:
            return 0


class InMemoryVectorStore(VectorStore):
    """内存向量存储（开发/测试用）"""

    def __init__(self, embedding_provider: "EmbeddingProvider"):
        self._embedder = embedding_provider
        self._collections: dict[str, List[dict]] = {}

    async def add_documents(self, chunks: List["DocumentChunk"], collection: str = "default"):
        if collection not in self._collections:
            self._collections[collection] = []

        texts = [c.text for c in chunks]
        embeddings = await self._embedder.embed_documents(texts)

        for chunk, embedding in zip(chunks, embeddings):
            self._collections[collection].append({
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {"source": chunk.source, "chunk_index": chunk.chunk_index},
            })

    async def search(self, query: str, collection: str = "default", top_k: int = 5, score_threshold: float = 0.7):
        import numpy as np
        if collection not in self._collections:
            return []

        query_emb = np.array(await self._embedder.embed_query(query))
        results = []

        for item in self._collections[collection]:
            doc_emb = np.array(item["embedding"])
            score = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
            if score >= score_threshold:
                results.append({"text": item["text"], "score": round(score, 4), "metadata": item["metadata"]})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def delete_collection(self, collection: str):
        self._collections.pop(collection, None)

    async def count(self, collection: str = "default") -> int:
        return len(self._collections.get(collection, []))


def create_vector_store(config: dict, embedding_provider: "EmbeddingProvider") -> VectorStore:
    """Factory function"""
    store_type = config.get("type", "chroma")

    if store_type == "chroma":
        return ChromaVectorStore(
            embedding_provider=embedding_provider,
            persist_dir=config.get("persist_dir", "./data/chroma"),
        )
    elif store_type == "memory":
        return InMemoryVectorStore(embedding_provider=embedding_provider)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
```

---

## 5. RAG 检索器 `rag/retriever.py`

```python
"""
RAG retriever —— 将用户查询转化为向量检索，返回相关文档片段。

使用流程：
1. Agent Core 检测到需要知识检索（通过 tool call 或自动触发）
2. Retriever.search() → 返回 top-k 相关文本块
3. 结果注入到 LLM prompt 的 context section
"""

from typing import List, Optional


class RAGRetriever:
    """RAG 检索器"""

    def __init__(self, vector_store: "VectorStore", collection: str = "default"):
        self._store = vector_store
        self._collection = collection

    async def search(
        self, query: str, top_k: int = 5, score_threshold: float = 0.7,
    ) -> List[dict]:
        """执行检索"""
        return await self._store.search(query, self._collection, top_k, score_threshold)

    def format_context(self, results: List[dict], separator: str = "\n\n---\n\n") -> str:
        """将检索结果格式化为 prompt context"""
        if not results:
            return ""

        formatted_chunks = []
        for i, r in enumerate(results, 1):
            source_info = r["metadata"].get("source", "unknown")
            chunk_idx = r["metadata"].get("chunk_index", "?")
            formatted_chunks.append(
                f"[来源 {i}: {source_info} (块 #{chunk_idx}), 相关度: {r['score']}]\n{r['text']}"
            )

        return separator.join(formatted_chunks)


class RAGTool:
    """
    作为 Tool 注册到 ToolRegistry，供 Agent 自主调用。

    LLM 可以通过 function calling 触发知识检索：
    {
      "name": "knowledge_search",
      "arguments": {"query": "...", "top_k": 5}
    }
    """

    name = "knowledge_search"
    description = "在知识库中搜索相关信息。适用于需要事实性回答、文档查询的场景。"

    def __init__(self, retriever: RAGRetriever):
        from tools.base import BaseTool, ToolParameter
        self._retriever = retriever

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询"},
                        "top_k": {"type": "integer", "default": 5, "description": "返回结果数量"},
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, query: str, top_k: int = 5) -> dict:
        results = await self._retriever.search(query, top_k=top_k)
        context = self._retriever.format_context(results)

        if not context:
            return {"content": "未在知识库中找到相关信息。"}

        return {
            "content": f"找到 {len(results)} 条相关结果：\n\n{context}",
            "results_count": len(results),
        }
```

---

## 6. System Prompt 注入策略

```python
"""
RAG context injection to system prompt。

两种模式：
1. 自动模式: Agent Core 在每次 LLM 调用前自动检索（基于用户消息）
2. 工具模式: LLM 自主决定何时调用 knowledge_search tool
"""


def inject_rag_context(system_prompt: str, rag_context: str) -> str:
    """将 RAG 检索结果注入到 system prompt"""
    if not rag_context.strip():
        return system_prompt

    context_section = f"""
## 参考知识（请基于以下信息回答，如与问题无关可忽略）：
{rag_context}
"""
    return system_prompt + context_section


class AutoRAGWrapper:
    """自动 RAG —— Agent Core 的中间件式包装"""

    def __init__(self, retriever: RAGRetriever, min_score: float = 0.75):
        self._retriever = retriever
        self._min_score = min_score

    async def enrich_prompt(self, system_prompt: str, user_message: str) -> str:
        """在 LLM 调用前自动检索并注入 context"""
        results = await self._retriever.search(user_message, score_threshold=self._min_score)

        if not results:
            return system_prompt

        rag_context = self._retriever.format_context(results)
        return inject_rag_context(system_prompt, rag_context)
```

---

## 7. YAML 配置 `rag` section

```yaml
# config/settings.yaml (新增)
rag:
  enabled: false              # 默认关闭，按需开启

  embedding:
    provider: "openai"        # openai | local
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-3-small"

  vector_store:
    type: "chroma"            # chroma | memory
    persist_dir: "./data/chroma"

  retrieval:
    top_k: 5                  # 每次检索返回的文档块数量
    score_threshold: 0.7      # 最低相关度阈值
    mode: "tool"              # tool (LLM自主调用) | auto (自动注入)

  chunking:
    strategy: "sliding_window"  # sliding_window | semantic
    chunk_size: 512             # 每块最大字符数
    overlap: 64                 # 重叠窗口大小
```

---

## 8. 数据流图

```
用户消息 → Agent Core
              │
              ├─ [auto mode] ─→ RAGRetriever.search()
              │                    │
              │                    ├─ EmbeddingProvider.embed_query()
              │                    │
              │                    └─ VectorStore.search()
              │                         │
              │                         ▼
              │                    format_context() → inject_rag_context(system_prompt)
              │                         │
              │                         ▼
              │                   LLM.chat(messages + enriched prompt)
              │
              └─ [tool mode] ─→ LLM 自主调用 knowledge_search tool
                                   │
                                   ├─ ToolRegistry.execute("knowledge_search")
                                   │    └─ RAGRetriever.search() → 结果回注 conversation
                                   │
                                   └─ LLM 基于检索结果生成最终回复
```

---

## 9. 设计总结

| 特性 | 实现方式 |
|------|---------|
| **Embedding** | ABC + Factory，OpenAI / Local sentence-transformers |
| **分块策略** | SlidingWindow（固定大小+重叠）/ Semantic（按标题/段落） |
| **向量存储** | ChromaDB（本地持久化）/ InMemory（开发测试） |
| **检索模式** | Tool 模式（LLM 自主调用）/ Auto 模式（自动注入 prompt） |
| **文档解析** | PDF、Markdown、HTML、TXT、DOCX 多格式支持 |
| **热配置** | YAML 控制启停、provider 切换、参数调优 |
