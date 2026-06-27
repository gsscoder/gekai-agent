# Locator cache (hybrid retrieval)

The `locate` pipeline uses a hybrid retriever to pre-seed the file locator with candidate paths before the LLM tool-call phase.

## How it works

1. **Lexical** (`find_candidates`): keyword overlap against `file_keywords` — fast, exact-token match.
2. **Semantic** (`find_semantic`): KNN on chunk embeddings via sqlite-vec vec0 — catches concept queries that don't match tokens.
3. **Fusion** (`find_hybrid`): Reciprocal Rank Fusion (RRF, c=60) over both result lists. Score = `Σ 1/(60 + rank)`. Top-k paths passed as `hint_paths` to the LLM locator.

## Why hybrid

Dense embeddings miss exact identifiers (function names, error codes, paths); keyword search misses concepts ("authentication" vs `auth_handler`). They fail in opposite directions — fusion is strictly better than either alone.

## Index build

`build_index` is called at workspace startup ("onboard" task). It chunks each file into 40-line windows (10-line overlap), embeds in batches via fastembed (bge-small-en-v1.5, 384-dim), and stores in the `file_chunks` vec0 table. Incremental: files with matching stat (size + mtime_ns) are skipped.

## Ponytail notes

- No ANN index — repo scale is thousands of chunks; SIMD brute-force KNN is sub-ms. Add when multi-repo scale is needed.
- Model is lazy-loaded on first query (~130 MB download on first run).
