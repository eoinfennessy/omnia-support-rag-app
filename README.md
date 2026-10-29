# Omnia Support - AI Chatbot

A multi-agent AI chatbot for customer support, combining RAG with hybrid search, a document-ingestion pipeline, and multi-turn conversational memory. A triage agent identifies intent, breaks complex requests into sub-queries, and routes them to specialist agents; MCP servers connect agents to order, refund, and inventory tools. An MLflow-backed evaluation framework tracks agent and RAG pipeline performance.

<img width="909" height="805" alt="Image" src="https://github.com/user-attachments/assets/28d9e6d4-bbe1-49d6-b89a-4461b028e05a" />

See [presentation/README.md](./presentation/README.md) for system architecture and pipeline flows.

----

# Getting Started

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - for local development
- [Docker](https://www.docker.com/) - optiona, for containerized workflows
- [Node.js](https://nodejs.org/) - optional, for building the SvelteKit frontend
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/setu-ibm/ai-assistant-eoinfennessy.git
   cd ai-assistant-eoinfennessy
   ```

2. Create a `.env` file in the project root with your OpenAI API key:

   ```bash
   cp .env.example .env
   # Edit .env and replace "sk-your-key-here" with your actual key
   ```

## Running the App

### Option A: Docker (recommended)

Start all services (ChromaDB, MLflow, ingestion, prompt register, and the app):

```bash
make up
```

The app will be available at [http://localhost:8080](http://localhost:8080).

### Option B: Local development

1. Start the infrastructure services (ChromaDB and MLflow):

   With docker:

   ```bash
   make infra-docker
   ```

   Or natively via uv:

   ```bash
   make infra-local
   ```

2. Run the ingestion pipeline to load the knowledge base into ChromaDB:

   ```bash
   make ingest
   ```

3. Build the frontend and start the app server:

   ```bash
   make app
   ```

   The app will be available at [http://localhost:8080](http://localhost:8080).

## Running Evaluation

1. Ensure ChromaDB and MLflow are running (see above).

2. Run the evaluation suite:

   With Docker:

   ```bash
   make eval-docker
   ```

   Or locally using uv:

   ```bash
   make eval-local
   ```

   View evaluation results in the MLflow UI at [http://localhost:5000](http://localhost:5000).

## Running Tests

```bash
make test
```

## Code Quality

```bash
make check    # Run all checks (lint, format, type-check, test)
make fix      # Auto-fix lint and formatting issues
```

# Self Assessment

## Assignment 1a

1. **Conversation Memory: Handle multi-turn conversations where context matters**
   - Completed - [X]
   - Very short writeup on completion:
   
   History is threaded through two parts of the RAG pipeline:

   1. the LLM query interpreter, which uses prior turns to rewrite follow-up questions into standalone retrieval and generation queries, and
   2. prompt rendering, which splices history between the system prompt and current turn so the LLM can maintain conversational context.

   An in-memory session store is implemented in the app. This tracks per-session chat history as developer-user-assistant triples.


2. **Hybrid Retrieval Strategy: Combine multiple retrieval approaches**
   - Semantic search for similar past issues - [X]
   - Keyword search for exact product names/order numbers - [X]
   - Metadata filtering (recency, category, satisfaction score) - [X]
   - Students must design and justify their hybrid approach: include a blurb on your justification

   Semantic and BM25 retrievers are combined using Reciprocal Rank Fusion, weighted by retriever type - a 6:4 semantic-to-keyword ratio worked well in testing. Each retriever has an independent threshold filter calibrated to its own score range to reject irrelevant results before fusion. An LLM query interpreter uses structured output to extract product categories from the user's question and apply them as `where` filters, so both retrievers search only the relevant slice of the collection. After fusion, multiplicative rescorers adjust ranking by document type relevance (also determined thru interpreted query), customer satisfaction, and resolution status. This favours resolved, high-satisfaction, contextually appropriate documents.


3. **Dynamic Context Selection: Intelligently choose how much context to include**
   - Not all retrieved documents are equally relevant
   - Must implement re-ranking or relevance scoring - [X]
   - Balance between context quality and token limits: include a blurb on the tradeoffs

   After retrieval, three multiplicative rescorers adjust chunk scores by document type relevance, customer satisfaction, and resolution status, then chunks are re-sorted by their adjusted scores.
   
   A `TokenBudgetFormatter` fills an n-token budget in score order, dropping lower-ranked chunks. A `LostInMiddleFormatter` then interleaves the surviving chunks so the highest-ranked items sit at the start and end of the context window, where LLMs attention is strongest (Liu et al., 2023). The overfetch pattern (3 more candidate docs before filtering) ensures enough quality chunks survive threshold and budget cuts.

4. **Fallback Handling: Gracefully handle cases where RAG doesn't help**
   - Detect when no relevant information exists - [X]
   - Provide helpful fallback responses - [X]
   - Flag for human escalation with reasoning - []

   Fallback operates at two levels. At the retrieval level, threshold filters on both retrievers reject low-confidence results - if every chunk is filtered out, the pipeline returns a set fallback response asking the user to rephrase, without calling the LLM. At the generation level, the system prompt instructs the model to say it can't answer the question instead of guessing when the retrieved context is insufficient. Human escalation flagging is not implemented.

## Assignment 2a

1. **Specialised Agent Architecture: Build at least 3 specialised agents** 
   - Triage Agent: Classify urgency, route to appropriate handler - [X]
   - Information Agent: Uses advanced RAG with query decomposition - [X]
   - Action Agent: Executes operations via tool calling - [X]
   - Agents must communicate and coordinate - [X]

2. **Query Decomposition & Multi-Step Reasoning: Handle complex requests** 
   - Must break down into sub-problems - [X]
   - Execute multiple retrieval operations - [X]
   - Synthesise comprehensive response - [X]

3. **Tool Integration: Connect to simulated backend systems (you'll provide mock APIs)**
   - At least 1 API connected - [X]
   - MCP used - [X]

4. **Advanced Retrieval Strategies: Implement at least 2 advanced techniques**
   - Query expansion/rewriting - [X]
   - Hypothetical document embeddings (HyDE) - []
   - Parent-child document chunking - []
   - Temporal/metadata boosting - [X]
   - Students choose and justify their approaches - [X]
      - Query expansion, rewriting and metadata filtering & boosting were included & configured because of improvements in evaluation metrics. See [elective/README.md](elective/README.md) for more detail.