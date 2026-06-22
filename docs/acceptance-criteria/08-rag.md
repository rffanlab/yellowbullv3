# RAG — 验收标准

## RAG-01: Document Ingestion — PDF

**优先级**: P0

- **Given** `RAGService` 已初始化（ChromaDB + Embedding）
- **When** 调用 `ingest(file_path="doc.pdf")`
- **Then** PDF 被解析为文本，按 chunk_size 切分；每个 chunk 生成 embedding 并存入 ChromaDB；返回 chunk_count
- **验证方式**: 集成测试（测试 PDF 文件）

## RAG-02: Document Ingestion — Markdown/Text

**优先级**: P0

- **Given** `RAGService` 已初始化
- **When** 调用 `ingest(file_path="doc.md")` / `ingest_text(text)`
- **Then** 文本按 chunk_size + chunk_overlap 切分；每个 chunk 包含 metadata（source、page、chunk_index）
- **验证方式**: 集成测试

## RAG-03: Chunking Strategy

**优先级**: P1

- **Given** 长文档（> chunk_size * 10）
- **When** 执行切分
- **Then** chunk 数量 = ceil(文本长度 / (chunk_size - overlap))；相邻 chunk 有 overlap 重叠内容；不切断句子中间（尽量按换行/句号边界）
- **验证方式**: 单元测试 — 断言 chunk 数量和边界

## RAG-04: Embedding Generation

**优先级**: P0

- **Given** `EmbeddingService` 配置了 provider（OpenAI / Ollama）
- **When** 调用 `embed(texts)`
- **Then** 返回 `list[list[float]]`，长度 = len(texts)；每个 vector 维度一致（如 1536 for text-embedding-3-small）
- **验证方式**: 单元测试（mock API）

## RAG-05: Vector Search — Top K

**优先级**: P0

- **Given** ChromaDB 中已索引 100 个 chunks
- **When** 调用 `search(query, top_k=5)`
- **Then** 返回 5 个结果，按 relevance score 降序排列；每个结果包含 text、source、score
- **验证方式**: 集成测试

## RAG-06: Vector Search — Score Threshold

**优先级**: P1

- **Given** ChromaDB 中已索引 chunks
- **When** 调用 `search(query, top_k=5, score_threshold=0.8)`
- **Then** 只返回 score >= 0.8 的结果；可能少于 5 条；无符合条件的结果返回空列表
- **验证方式**: 集成测试

## RAG-07: Vector Search — Metadata Filter

**优先级**: P1

- **Given** ChromaDB 中 chunks 带有 source metadata
- **When** 调用 `search(query, where={"source": "doc.pdf"})`
- **Then** 只返回指定 source 的 chunks；过滤在向量搜索阶段执行（非后过滤）
- **验证方式**: 集成测试

## RAG-08: Context Builder — 组装 Prompt

**优先级**: P1

- **Given** search 返回 5 个相关 chunks
- **When** 调用 `build_context(query, results)`
- **Then** 返回格式化字符串，包含每个 chunk 的文本和来源引用；总长度不超过 max_tokens 限制
- **验证方式**: 单元测试

## RAG-09: RAG Tool — search_knowledge_base

**优先级**: P1

- **Given** `search_knowledge_base` 工具已注册到 ToolRegistry
- **When** AgentCore 调用该工具（`{"query": "xxx"}`）
- **Then** 返回 ToolResult，content 包含格式化后的上下文片段；无结果时返回明确提示
- **验证方式**: 集成测试

## RAG-10: Document Deletion

**优先级**: P2

- **Given** ChromaDB 中已索引多个文档的 chunks
- **When** 调用 `delete_document(source="doc.pdf")`
- **Then** 该文档所有 chunks 从向量库移除；search 不再返回相关结果
- **验证方式**: 集成测试

## RAG-11: Embedding Cache

**优先级**: P2

- **Given** 相同文本被多次请求 embedding
- **When** 调用 `embed(texts)`
- **Then** 第二次及后续请求命中缓存，不重复调用 LLM API；返回结果一致
- **验证方式**: 单元测试 — mock API + 断言调用次数

## RAG-12: Batch Ingestion

**优先级**: P2

- **Given** 目录包含多个文档文件
- **When** 调用 `ingest_directory(path)`
- **Then** 所有支持的格式（.pdf, .md, .txt）被处理；返回统计信息（总 chunk 数、成功/失败文件数）
- **验证方式**: 集成测试
