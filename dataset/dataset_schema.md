# Omnia Retail Ltd — Customer Support Knowledge Base
## Dataset Documentation (README)

**Version:** 1.0  
**Created:** April 2024  
**Module:** AI Engineering — Assignment 1 (RAG System Development)  
**Fictional Entity:** Omnia Retail Ltd (Irish electronics retailer)

---

## Overview

This dataset is a synthetic knowledge base representing the customer support documentation of **Omnia Retail Ltd**, a fictional Irish electronics retailer selling laptops, monitors, accessories, and storage products. It is designed for use in building and evaluating RAG (Retrieval Augmented Generation) systems.

The dataset is structured to provide realistic, varied, and semantically rich documents that reward good retrieval strategies and expose weaknesses in naive keyword-matching approaches.

---

## Dataset Statistics

| Type | Count | ID Range |
|---|---|---|
| Support Tickets | 40 | ticket_001 – ticket_040 |
| Policies | 10 | policy_001 – policy_010 |
| FAQs | 25 | faq_001 – faq_025 |
| **Total** | **75** | — |

---

## Document Schema

Each document in the `documents` array follows this schema:

```json
{
  "document_id": "ticket_001",
  "type": "support_ticket | policy | faq",
  "category": "string",
  "date": "YYYY-MM-DD",
  "resolution_status": "resolved | active | pending | null",
  "customer_satisfaction": 4.5,
  "product_category": "laptops | monitors | accessories | storage | all | none",
  "priority": "low | medium | high | null",
  "content": "The actual text content for retrieval..."
}
```

### Field Notes

| Field | Applies To | Description |
|---|---|---|
| `document_id` | All | Unique identifier |
| `type` | All | Document type: `support_ticket`, `policy`, `faq` |
| `category` | All | Topic category (see categories below) |
| `date` | All | Document creation / ticket date (YYYY-MM-DD) |
| `resolution_status` | Tickets only | `resolved`, `active`, `pending`; `null` for policies/FAQs |
| `customer_satisfaction` | Tickets only | Float 1.0–5.0; `null` for unresolved tickets and non-ticket documents |
| `product_category` | All | Primary product area; `all` for cross-category; `none` for non-product docs |
| `priority` | Tickets only | `low`, `medium`, `high`; `null` for non-ticket documents |
| `content` | All | Full text content used for retrieval (200–800 words) |

---

## Document Categories

### Support Ticket Categories
- `delivery_issues` — lost parcels, misdelivery, transit damage
- `defects` — hardware faults, manufacturing defects
- `returns` — change-of-mind returns, DOA replacements
- `technical_support` — troubleshooting, diagnostics, driver issues
- `pre_sales` — product recommendations, buying advice
- `compatibility` — port types, RAM compatibility, ecosystem fit
- `installation_help` — setup, upgrade guidance
- `product_comparison` — side-by-side product advice
- `warranty_claims` — manufacturer warranty, statutory rights
- `bulk_orders` — business procurement, volume pricing

### Policy Categories
- `returns_and_refunds` (policy_001)
- `warranty` (policy_002)
- `privacy_data_protection` (policy_003)
- `delivery` (policy_004)
- `price_matching` (policy_005)
- `business_customers` (policy_006)
- `environmental_sustainability` (policy_007)
- `product_reviews` (policy_008)
- `terms_conditions` (policy_009)
- `acceptable_use` (policy_010)

### FAQ Categories
- `product_questions` — hardware specs, buying guidance
- `technical_troubleshooting` — step-by-step diagnostic guides
- `account_management` — account settings, data management
- `shipping_info` — delivery options, lost orders
- `warranty_questions` — warranty coverage, claims process

---

## Geographic and Regulatory Context

All documents are set in the **Republic of Ireland** (and Northern Ireland where applicable) with the following real-world context woven in:

- **Currency:** Euro (€)
- **Consumer law:** Consumer Rights Act 2022 (Ireland); EU Sale of Goods Directive 2019/771/EU; EU Consumer Rights Directive 2011/83/EU
- **Data protection:** GDPR (Regulation (EU) 2016/679); Data Protection Acts 1988–2018
- **Environmental:** WEEE Ireland scheme; EU WEEE Directive 2012/19/EU; RoHS Directive
- **Carriers:** An Post (standard), DPD (express)
- **Geography:** Dublin, Cork, Galway, Limerick, Waterford, Donegal, Kerry, and other Irish locations
- **Irish language:** One ticket (ticket_032) includes a brief Irish-language customer quote

---

## Test Queries

The dataset includes 10 example test queries with ground truth expected document IDs. These are stored in the `test_queries` array of the main JSON file.

Each test query has:
- `query_id` — unique identifier (TQ-001 to TQ-010)
- `query` — natural language query as a user might type it
- `expected_document_ids` — list of document IDs that should be retrieved
- `category` — topic area
- `notes` — explanation of why those documents are relevant

### Test Query Summary

| ID | Query (abbreviated) | Expected docs |
|---|---|---|
| TQ-001 | Return policy for change of mind | policy_001, ticket_018, ticket_028 |
| TQ-002 | Laptop keeps shutting down unexpectedly | ticket_025, ticket_040, faq_008, faq_016 |
| TQ-003 | How much RAM for video editing? | faq_006, faq_010, ticket_020 |
| TQ-004 | Order shows delivered but not received | ticket_016, faq_018 |
| TQ-005 | Warranty rights at 10 months | faq_009, ticket_019, ticket_021, faq_017 |
| TQ-006 | Bulk monitor purchase / business pricing | policy_006, ticket_039, ticket_023 |
| TQ-007 | NVMe vs SATA SSD difference | faq_012, ticket_024, ticket_038 |
| TQ-008 | Home office setup on a budget | faq_015, ticket_026, faq_011 |
| TQ-009 | External hard drive clicking + not detected | faq_020, ticket_021 |
| TQ-010 | WEEE recycling / old electronics disposal | policy_007 |

---

## RAG System Design Notes

### Chunking Recommendations
- **Support tickets** have a natural structure (Customer → Investigation → Resolution). Consider chunking at section boundaries using double newline as a separator.
- **Policies** are long-form. Chunk by numbered section headings for best retrieval granularity.
- **FAQs** are already well-structured as standalone question-answer pairs. Treat each FAQ as a single chunk.
- Recommended chunk size: 300–500 tokens with 50-token overlap.

### Embedding Considerations
- Metadata fields (`type`, `category`, `product_category`, `priority`) can be prepended to chunks before embedding to enrich semantic representation.
- The dataset rewards semantic search over keyword search — many queries use different vocabulary from the documents (e.g. "order not received" vs. "misdelivery").

### Evaluation Approach
- Use the `test_queries` array to evaluate retrieval precision and recall.
- For each query, retrieve top-k documents (k=3 or k=5) and check whether `expected_document_ids` appear in the retrieved set.
- A well-tuned RAG system should achieve >80% recall@3 on this dataset.

---

## Intended Use

This dataset is provided for educational use within the AI Engineering module at SETU. It must not be used for commercial purposes. The fictional company, characters, order numbers, and scenarios are entirely invented. Any resemblance to real persons, companies, or events is coincidental.

---

## Contact

For questions about this dataset, contact your module lecturer or raise a query via the course Moodle forum.
