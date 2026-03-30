# TODO - 待优化项

## 排序与检索优化

### 1. 集成外部 Reranker
当前只有 Vespa 的两阶段排序和 RRF 合并。需要集成 cross-encoder（如 ms-marco-MiniLM、BGE-reranker）或商业 API（如 Cohere Rerank）进行重排，以提升检索相关性。

---

### 2. 自动计算 aggregated_chunk_boost_factor
字段已预留但无自动计算逻辑，默认固定为 1.0。需要在 `SemanticChunker` 中根据内容长度、信息密度等特征自动评估质量，低质量设置 < 1.0，高质量设置 > 1.0。

---

### 3. 启用轻量级重排
`rerank_by_relevance()` 函数已实现但从未被调用。可在 `RetrievalPipeline` 中可选启用，作为无需额外模型的低成本重排方案。

---

## 优先级

1. **集成外部 Reranker**（高复杂度，高收益）
2. **自动计算 boost**（中低复杂度，中收益）
3. **启用轻量级重排**（低复杂度，低收益）
