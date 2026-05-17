# A/B Testing Report

An A/B testing framework has been established using MLflow. This is briefly described below before going into the results

## Run configuration

Experiment runs are configured using a [config yaml file](packages/evaluation/eval_config.example.yaml). This config includes settings for configuring the RAG pipeline, including...

- retrieval - hybrid, semantic, keyword, weights, post-retrieval filters, RFF config, n-results etc.,
- prompt version - an ID or alias of a prompt registered in MLflow,
- the model to use,
- the query interpreter to use,
- the rescorers to use and their config,
- metadata filtering options,
- etc.

These configurations are logged as params alongside each run, providing observability to link config params with the run results, and alowing users to filter / group / sort runs by param values through the dashboard or API.

## Scorers

Each evaluation run applies 7 scorers. 5 LLM-as-judge scorers and 2 deterministic retrieval metrics:

- Correctness - whether the response supports the facts in the expected response
- Relevance to query - whether the response directly addresses the user's query without deviating into unrelated topics
- Retrieval groundedness - whether the response is grounded in the retrieved context (i.e. no made-up facts)
- Retrieval relevance - whether each retrieved chunk is relevant to the query, and an aggregate score of all chunks
- Retrieval sufficiency - whether the retrieved documents contain enough information to produce the expected answer
- Retrieval recall - percentage of expected document IDs found in the retrieved results
- Mean reciprocal rank - reciprocal of the position of the first relevant document in the reranked results

## Results

### Comparing query interpretation prompt versions

This comparison tests whether synonym expansion in the `retrieval_query` field of the query interpretation prompt improves retrieval and answer quality.

Prompt v8 instructs the LLM to expand the query with key synonyms and related terms (e.g. "return policy" becomes "return policy refund period cooling-off consumer rights"), while v7 instructs the LLM to use the clean rewritten query as-is for both `query` and `retrieval_query`. All other variables remain constant.

| Metric | v8 (synonym expansion) | v7 (no expansion) |
|---|---|---|
| Correctness | 0.897 | 0.828 |
| Relevance to Query | 0.966 | 0.966 |
| Retrieval Groundedness | 0.828 | 0.793 |
| Retrieval Relevance | 0.466 | 0.346 |
| Retrieval Sufficiency | 0.897 | 0.793 |
| Retrieval Recall | 0.901 | 0.814 |
| MRR | 0.872 | 0.756 |

Synonym expansion in `retrieval_query` delivers consistent improvements across all retrieval metrics: recall improves by 9 percentage points and MRR by 12 points, indicating that expanded queries help the retriever both find more relevant documents and rank them higher. These retrieval gains cascade into downstream answer quality, with correctness improving by 7 points and retrieval sufficiency by 10 points. The only metric unaffected is relevance to query, which remains at 0.966 for both. This suggesting that both prompt versions produce on-topic responses regardless of retrieval quality.

### Comparing retrieval types

This comparison tests hybrid retrieval (semantic + BM25 via Reciprocal Rank Fusion with 0.6/0.4 weighting) against semantic-only retrieval. All other variables remain constant.

| Metric | Hybrid | Semantic-only |
|---|---|---|
| Correctness | 0.897 | 0.897 |
| Relevance to Query | 0.966 | 0.966 |
| Retrieval Groundedness | 0.828 | 0.828 |
| Retrieval Relevance | 0.466 | 0.484 |
| Retrieval Sufficiency | 0.897 | 0.828 |
| Retrieval Recall | 0.901 | 0.862 |
| MRR | 0.872 | 0.903 |

The two strategies present trade-off between recall and precision. Hybrid retrieval achieves higher recall and sufficiency, meaning BM25 catches documents that semantic search misses. However, semantic-only retrieval achieves a higher MRR and retrieval relevance, indicating it ranks the single best document higher and introduces less noise. Despite retrieval differences, downstream correctness and groundedness are identical, suggesting that the LLM produces similar answer accuracy for both retrieval strategies.

### Impact of LLM query interpretation

This comparison tests the pre-retrieval LLM query interpretation step. With interpretation enabled, the LLM rewrites the query, expands it with synonyms, identifies relevant product categories, and scores document type relevance to enable metadata filtering and document-type rescoring. Without interpretation, the raw user query is passed directly to the retriever with no metadata filtering. All other variables remain constant.

| Metric | With interpretation | Without interpretation |
|---|---|---|
| Correctness | 0.897 | 0.724 |
| Relevance to Query | 0.966 | 0.966 |
| Retrieval Groundedness | 0.828 | 0.828 |
| Retrieval Relevance | 0.466 | 0.368 |
| Retrieval Sufficiency | 0.897 | 0.793 |
| Retrieval Recall | 0.901 | 0.833 |
| MRR | 0.872 | 0.772 |

Query interpretation provides the largest single improvement of all three comparisons, with correctness jumping by 17 percentage points, MRR by 10 points and sufficiency by 10 points. The gains come from three effects: query rewriting disambiguates vague references, synonym expansion improves recall, and metadata filtering narrows retrieval to relevant document types. Groundedness and relevance to query are unaffected, meaning the improvement is purely in retrieval quality translating to more accurate answers, not in generation behaviour.