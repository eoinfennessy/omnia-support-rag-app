# How to Use the Dataset in Your Assignment

## Overview

This document explains how the **Omnia Retail Knowledge Base** dataset connects to your Assignment 1 requirements. You've been provided with a starter dataset so you can focus on building a sophisticated RAG system rather than spending time on data collection.

---

## What You've Been Given

### 1. Knowledge Base (`omnia_retail_knowledge_base.json`)
A realistic customer support knowledge base containing **75 documents**:
- **40 support tickets**: Real customer scenarios with resolutions
- **10 company policies**: Returns, warranty, shipping, privacy, business customers, etc.
- **25 FAQs**: Common questions about products, technical issues, and processes

### 2. Dataset Schema (dataset_schema.md)
Complete documentation of the metadata structure, field meanings, and possible values.

### 3. This Usage Guide
How to apply the dataset to meet assignment requirements.

---

## Understanding the Assignment Requirements

**Assignment 1** asks you to build an **Intelligent Customer Support Knowledge Assistant** - a RAG system that can:
- Answer customer questions using the knowledge base
- Maintain conversation context across multiple turns
- Use hybrid retrieval (combining multiple retrieval strategies)
- Select relevant context intelligently
- Handle cases where the answer isn't in the knowledge base

The dataset is designed to support all these requirements.

---

## Part 1: Loading and Understanding the Data

### Step 1: Examine the Dataset Structure

Before writing any code, open the JSON file and understand what you're working with:

**Each document has:**
- **`content`**: The actual text that will be embedded and retrieved
- **`metadata`**: Structured information about the document (type, category, dates, satisfaction scores, etc.)

**Key insight**: The `content` field is what your RAG system will search through semantically. The metadata fields enable intelligent filtering and ranking.

### Step 2: Understand Document Types

**Support Tickets (`type: "support_ticket"`)**
- Real customer scenarios with problem → investigation → resolution
- Include customer satisfaction scores (1.0-5.0)
- Show how issues were actually resolved
- Use case: Great for "how do I fix X" questions

**Policies (`type: "policy"`)**
- Formal company policies (returns, warranty, shipping, etc.)
- Authoritative source for "what is your policy on X" questions
- Longer, comprehensive documents
- Use case: Direct policy questions should retrieve these

**FAQs (`type: "faq"`)**
- Question-and-answer format
- Cover common questions and how-to guidance
- Bridge between tickets (specific cases) and policies (formal rules)
- Use case: General product and process questions

### Step 3: Study the Metadata Schema

The metadata schema document explains every field. **Pay special attention to:**

- **`type`**: Filter by document type (ticket/policy/faq)
- **`category`**: Group related content (delivery_issue, product_defect, technical_support, etc.)
- **`product_category`**: Filter by product type (laptops, monitors, accessories, storage)
- **`customer_satisfaction`**: Quality indicator (4.5+ means highly successful resolution)
- **`date`**: Temporal filtering (prefer recent information)
- **`priority`**: Indicates severity of issues

**Why this matters**: These metadata fields are your tools for implementing hybrid retrieval and intelligent context selection.

---

## Part 2: Mapping Dataset to Assignment Requirements

### Deliverable 1A: Multi-turn Conversational RAG

#### **Conversation Memory**

The dataset supports multi-turn conversations naturally:

**Example conversation flow:**
1. User: "I want to return my laptop"
   - System retrieves: Return policy (policy_001)
   - Answer: Explains 30-day return window

2. User: "I bought it 45 days ago, what can I do?"
   - System remembers context (return discussion, laptop)
   - Retrieves: Related support tickets showing similar scenarios
   - Answer: Explains policy limitation, suggests alternatives

3. User: "Can I get a discount on a different model instead?"
   - System maintains full conversation context
   - Retrieves: Past tickets showing goodwill gestures
   - Answer: Explains possibilities based on precedent

**Dataset usage**: Each support ticket shows complete scenarios, teaching your system how similar situations were handled. The conversation memory lets you reference previous exchanges.

#### **Hybrid Retrieval**

This is where the metadata schema becomes critical. Hybrid retrieval means combining multiple strategies:

**Strategy 1: Semantic Search**
- Embed the `content` field
- Find documents semantically similar to the query
- This is your baseline retrieval

**Strategy 2: Metadata Filtering**
- Use `type` to filter document types (policy questions → type="policy")
- Use `product_category` to filter by product (laptop questions → product_category="laptops")
- Use `category` to find related issues (delivery problems → category="delivery_issue")

**Strategy 3: Quality Boosting**
- Prefer high `customer_satisfaction` scores (proven solutions)
- Boost recent documents using `date` field
- Prioritize high `priority` tickets (common/important issues)

**Strategy 4: Keyword Matching**
- Combine with semantic search for specific terms
- Product names, order numbers, technical specifications

**Your task**: Implement at least 2-3 of these strategies and combine their results intelligently.

#### **Context Selection**

Not all retrieved documents are equally relevant. The dataset helps you practice context selection:

**Scenario**: User asks "My battery drains in 2 hours"

**Retrieved documents might include:**
- ticket_002: Dell XPS battery issue (HIGHLY RELEVANT - exact scenario)
- faq_007: Battery troubleshooting guide (RELEVANT - general guidance)
- policy_002: Warranty policy (SOMEWHAT RELEVANT - mentions battery coverage)
- ticket_010: Keyboard issue (IRRELEVANT - wrong problem)

**Your task**: Select which documents to pass to the LLM. Consider:
- Relevance score (semantic similarity)
- Document type (FAQs and exact-match tickets > tangentially related policies)
- Metadata alignment (battery question → laptop category)
- Token budget (can't send everything)

**Dataset feature**: The variety of document types and metadata lets you practice discriminating between highly relevant, somewhat relevant, and irrelevant content.

#### **Fallback Handling**

The dataset is **intentionally incomplete** - not every possible question has an answer. This forces you to implement fallback strategies.

**Questions with NO direct answer in the dataset:**
- "Do you sell gaming consoles?" (We only sell laptops/monitors/accessories/storage)
- "What's your store hours?" (We're online-only)
- "Can I trade in my PlayStation?" (No trade-in program mentioned)

**Questions with PARTIAL answers:**
- "My laptop won't turn on" (We have battery issues, but not comprehensive troubleshooting)
- "How do I upgrade my RAM?" (We mention RAM but no installation guides)

**Your fallback strategies should:**
1. Detect when retrieved content doesn't answer the question
2. Admit knowledge gaps honestly
3. Offer related information that IS in the knowledge base
4. Guide users appropriately (contact support, check manufacturer resources, etc.)

**Dataset feature**: The gaps are intentional - use metadata to help detect when your retrieval is weak (low similarity scores, no metadata alignment).

---

### Deliverable 1B: Advanced Components (Choose 1)

#### **Option 1: Cost-Performance Optimization**

**Dataset usage:**
- The 75 documents give you a realistic corpus size
- Metadata enables cheaper "pre-filtering" before expensive embedding/LLM calls
- Compare strategies: "retrieve everything → LLM" vs "metadata filter → semantic search → LLM"

**What to measure:**
- Query cost (API calls to embeddings and LLM)
- Retrieval quality (precision/recall)
- Response time
- Cost per query

**Optimization experiments:**
- Filter by `type` before semantic search (reduces embedding calls)
- Cache common queries using `document_id` references
- Adjust retrieval depth (k=5 vs k=10 vs k=20)

#### **Option 2: Business Metrics Dashboard**

**Dataset usage:**
- Customer satisfaction scores in support tickets
- Document types and categories for classification
- Product categories for segmentation
- Dates for temporal analysis

**Metrics to track:**
- Query resolution rate (answered vs fallback)
- Average customer satisfaction of retrieved solutions
- Coverage by product category (laptops 60%, monitors 25%, etc.)
- Most common query categories
- Response quality over time

**The dataset provides:**
- Ground truth satisfaction scores to validate your system
- Rich metadata for multi-dimensional analysis
- Realistic distribution of content types

#### **Option 3: A/B Testing Framework**

**Dataset usage:**
- Test different retrieval strategies on the same queries
- Use metadata to create control groups

**Experiments to run:**
- Strategy A: Pure semantic search
- Strategy B: Semantic + metadata filtering
- Strategy C: Semantic + metadata + quality boosting

**Evaluation using dataset:**
- Create test queries with expected `document_id` results
- Compare which strategy retrieves correct documents
- Measure precision, recall, and user satisfaction proxies

**The dataset enables:**
- Repeatable experiments (same knowledge base)
- Ground truth annotations (you know which documents should match)
- Metadata-rich evaluation (not just "did it retrieve something")

#### **Option 4: Edge Case Documentation**

**Dataset usage:**
- Identify query patterns the dataset handles poorly
- Document why certain questions fail
- Propose solutions using metadata

**Edge cases present in dataset:**
- **Ambiguous queries**: "How long does shipping take?" (depends on location - Zone 1/2/3)
- **Multi-document answers**: "What's your return and warranty policy?" (needs 2 separate policies)
- **Contextual dependencies**: "Can I return this?" (need to know purchase date from conversation)
- **Out-of-scope**: Questions about products/services not offered
- **Conflicting information**: Different policies for B2B vs retail customers

**Your documentation should:**
- Catalog these edge cases
- Explain how your system currently handles them
- Propose improvements using metadata (e.g., detect ambiguity from multiple category matches)

---

## Part 3: Creating Your Test Queries

The assignment requires you to submit **30 test queries with ground truth**. Here's how to use the dataset to create them:

### Types of Test Queries to Create

**1. Direct Matches** (should be easy)
- Query maps cleanly to one document
- Example: "What's your return policy?" → policy_001

**2. Multi-Document Synthesis** (moderate difficulty)
- Answer requires combining multiple documents
- Example: "My monitor arrived broken, what do I do?" → policy_001 (returns) + policy_002 (DOA warranty) + ticket_004 (damaged delivery example)

**3. Conversational Context** (tests memory)
- Multi-turn conversations requiring context
- Example: Turn 1: "I bought a laptop" → Turn 2: "How do I return it?" (needs to remember "laptop")

**4. Metadata-Dependent** (tests hybrid retrieval)
- Query where metadata filtering significantly improves results
- Example: "Laptop battery issues" → should prioritize product_category="laptops" + category="technical_support"

**5. Ambiguous/Edge Cases** (tests fallback)
- Questions with no clear answer or multiple interpretations
- Example: "What monitors do you recommend?" (no product catalog in dataset, only support content)

### Ground Truth Annotations

For each test query, document:
- **Expected document IDs**: Which documents should be retrieved (e.g., ["policy_001", "ticket_005"])
- **Expected answer elements**: Key facts that must appear in the response
- **Conversation context**: If multi-turn, what was discussed previously
- **Metadata expectations**: What filters should activate (type="policy", product_category="laptops")

**Why this matters**: You need objective criteria to evaluate your RAG system. "It gave an answer" isn't enough - did it retrieve the RIGHT documents and include the RIGHT information?

---

## Part 4: Evaluation Strategy

### Using the Dataset for Evaluation

**Retrieval Quality:**
- **Precision**: Of the documents retrieved, how many were relevant?
- **Recall**: Of the relevant documents in the dataset, how many were retrieved?
- **MRR (Mean Reciprocal Rank)**: Where did the most relevant document appear in results?

**Answer Quality:**
- Does the answer include expected information from ground truth?
- Are customer satisfaction scores of retrieved tickets high (4.0+)?
- Did it cite the correct document types (policy for policy questions)?

**Conversation Quality:**
- Does it maintain context across turns?
- Does it reference previous exchanges appropriately?
- Does it handle follow-up questions correctly?

**Fallback Quality:**
- Does it gracefully handle out-of-scope questions?
- Does it admit when information isn't in the knowledge base?
- Does it offer reasonable alternatives?

### Leveraging Metadata for Evaluation

The rich metadata lets you evaluate more than just "did it find something":

- **Type alignment**: Policy questions should retrieve type="policy" documents
- **Category alignment**: Delivery issues should retrieve category="delivery_issue" documents
- **Quality indicators**: Solutions should come from high customer_satisfaction tickets
- **Temporal relevance**: Recent issues should weight date appropriately

---

## Part 5: Common Pitfalls and How the Dataset Helps

### Pitfall 1: Ignoring Metadata
**Problem**: Only using semantic search, ignoring the rich metadata.
**Dataset helps**: Metadata enables hybrid retrieval - use it! A policy question semantically similar to a support ticket is still wrong if you ignore `type`.

### Pitfall 2: Poor Context Selection
**Problem**: Sending too many irrelevant documents to the LLM.
**Dataset helps**: Practice discriminating. Not everything retrieved is worth passing to the LLM. Use metadata to filter.

### Pitfall 3: No Fallback Strategy
**Problem**: Hallucinating answers when knowledge base lacks information.
**Dataset helps**: The intentional gaps force you to detect and handle missing information.

### Pitfall 4: Forgetting Conversation Context
**Problem**: Treating each query as independent.
**Dataset helps**: Many support tickets show multi-step scenarios. Study how context matters.

### Pitfall 5: Weak Test Queries
**Problem**: Creating test queries that are too easy or don't test edge cases.
**Dataset helps**: The variety of document types and scenarios gives you templates for comprehensive testing.

---

## Key Takeaways

1. **The dataset is realistic**: It has the messiness of real customer support - gaps, ambiguities, varying quality. This is intentional.

2. **Metadata is not optional**: The schema documentation exists for a reason. Hybrid retrieval requires using metadata intelligently.

3. **Think like a customer support system**: Your RAG system is answering real customer questions. Would a human support agent send all this retrieved information, or select the most relevant parts?

4. **Test comprehensively**: The 75 documents support many different query types. Your 30 test queries should cover this diversity.

5. **Document everything**: Your README and evaluation reports should explain your design choices, especially around hybrid retrieval and context selection.

6. **Embrace the gaps**: The dataset doesn't have all answers. Your fallback strategy is as important as your retrieval strategy.

---

## Questions to Guide Your Implementation

As you work through the assignment, keep asking:

- **Retrieval**: "Am I using all available metadata, or just semantic search?"
- **Quality**: "Am I preferring high-satisfaction solutions over low-quality ones?"
- **Context**: "Does my system remember what we discussed two questions ago?"
- **Relevance**: "Am I sending irrelevant documents to the LLM, wasting tokens and degrading answers?"
- **Fallback**: "What happens when the knowledge base doesn't have the answer?"
- **Evaluation**: "How do I know my system is working well? What metrics prove it?"

---

## Final Note

This dataset is a **teaching tool**. It's designed to let you practice building a production-quality RAG system without the overhead of data collection. The 75 documents are sufficient to demonstrate all required capabilities while being small enough to iterate quickly.

Your job is to show you can build an intelligent system that uses this data effectively - not just semantically, but leveraging the rich metadata structure to provide accurate, relevant, context-aware customer support.

Focus on **quality over quantity**. A well-implemented hybrid retrieval system with thoughtful context selection on 75 documents is more impressive than a naive semantic search system on 1000 documents.

**Good luck with your assignment!**
