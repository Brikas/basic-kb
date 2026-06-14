# RAG Query Writing Guide
## Semantic Search over Meeting Transcripts & Website Copy

System: ChromaDB + `bge-small-en-v1.5` (FastEmbed/ONNX) + multi-query merge by score + optional Jina reranker-v3.

---

## 1. The core principle (model-agnostic)

Embedding search matches **meaning and distribution**, not keywords. Every dense retrieval model — MiniLM, BGE, E5, Nomic — shares one fundamental constraint: **your query occupies a different region of embedding space than your documents** unless you write it to look like what you're looking for.

Documents in this system are spoken conversation snippets or marketing copy — statements and sentences. Queries that look like statements retrieve them reliably. Queries that look like keywords or questions do not.

**The single rule: write queries as statements, not search boxes.**

| Weak | Strong |
|---|---|
| `pricing objection` | `customer said the price was too high` |
| `churn cancel` | `customer mentioned they might not renew` |
| `webinar software marketing` | `punchy headline for a webinar platform landing page` |
| `integration problem` | `their team couldn't connect the API on time` |

This holds regardless of which embedding model is running underneath.

---

## 2. Why this matters more with BGE than with MiniLM

`all-MiniLM-L6-v2` (the old model) was trained on symmetric sentence pairs — query and document were expected to look similar in form. It tolerated keyword-style queries.

`bge-small-en-v1.5` (the current model) was trained for **asymmetric retrieval** with hard negative mining: during training, it was explicitly penalized for matching queries to documents that are *similar-but-wrong*. This makes it sharper, but also means vague keyword queries misfire more visibly — BGE's precision advantage works against you when your query is low-signal.

**Practical implication:** BGE rewards specificity. A well-formed statement retrieves much more precisely. A vague fragment may pull noise that MiniLM would have let through unnoticed.

**Note on instruction prefixes:** BGE models support a query instruction prefix (`"Represent this sentence for searching relevant passages: "`). This is handled by the pipeline's embedding layer — you never add it to query text manually. BGE v1.5 was specifically improved to work without it (slight degradation only per BAAI docs).

**Token limit:** 512 tokens. Optimal at 1–3 sentences. Queries longer than that get diluted embeddings.

---

## 3. HyDE: the most powerful single technique

HyDE (Hypothetical Document Embedding) is the best-evidenced query technique in the literature (arxiv 2212.10496, consistently replicated).

Instead of writing a query, write a **hypothetical document snippet** — the kind of text that would appear in your corpus if the answer were there.

**Why it works:** Dense models encode your query into embedding space. If your query is a question and your documents are statements, they live in different neighborhoods. A hypothetical statement-answer lives in the same neighborhood as the real documents. HyDE closes this gap without fine-tuning.

**For transcripts:** Imagine the actual spoken sentence.
```
# Instead of:
"did customers ask about pricing"

# Write:
"so we had a conversation about the cost and the customer said it was more than they expected"
```

**For website copy:** Imagine the actual headline or paragraph.
```
# Instead of:
"webinar software value proposition"

# Write:
"Turn every webinar into a sales asset your team can actually use"
```

**Important tradeoff:** HyDE adds latency (requires an LLM call to generate the hypothesis). When speed matters or the query is already well-formed, skip it and write a good statement directly. HyDE also adds hallucination risk: the generated hypothesis may introduce false assumptions that pull wrong documents.

---

## 4. Multi-query for recall

A single query hits one neighborhood. Multi-query expands coverage by approaching the same topic from multiple semantic angles, then merging by score.

**When to use:**
- High-recall research tasks (not spot-checks)
- Topics with multiple synonyms or framings
- You're unsure how the content was expressed

**Four angles to generate variants:**

**Perspective shift** — whose words?
- `customer explained why they were hesitant` → `the sales rep acknowledged the concern about price`

**Abstraction level** — specific vs. general
- `customer said the dashboard was slow` → `performance was flagged as a concern`

**Outcome framing** — what happened vs. what was said
- `customer asked for a discount` → `deal stalled on budget approval`

**Vocabulary / register**
- `churn risk` → `customer said they might cancel` → `they were considering switching to a competitor`

**Practical example — "What delayed deals in Q1?"**
```
customer said they needed more time before committing
deal was waiting for internal approval
budget hadn't been confirmed yet
they wanted to run a proof of concept first
there were concerns that slowed the process down
```

Fetch top-5 per query, merge, deduplicate, keep top-10 overall. The reranker handles final ordering if enabled.

---

## 5. Step-back prompting and decomposition

**Step-back:** Before writing specific queries, ask: *What broader concept contains this?*

Specific: `customer asked for a free trial`
Step-back: add `evaluation process was discussed` and `they wanted to test before buying`

**Decompose complex questions into sub-queries:**

Question: *"Why did we lose deals to Competitor X in H1?"*
→ `customer mentioned switching to another vendor`
→ `competitor was brought up as an alternative`
→ `customer decided not to move forward with us`
→ `reasons given for ending the evaluation`
→ `they chose a different solution`

Run each as a separate query. Merge results. This is more reliable than a single combined query — complex questions have multiple semantic components that land in different document neighborhoods.

---

## 6. When to split vs. combine queries

**Split when:** the question has two distinct components ("pricing AND onboarding friction") — one will dominate and bury the other.

**Combine when:** the relationship between concepts is what you're looking for ("implementation timeline was blocking the deal closed the same week").

---

## 7. Transcript-specific tips

Transcripts are spoken, informal, and messy. Match that register.

**Use colloquial phrasing.** Say `"they said they don't have budget right now"` not `"budget constraints were cited."` The transcript captured the spoken version.

**Query both sides of the conversation.** A pricing objection might be stored as the customer's words OR the AE's summary. Run both:
- `customer said the price was too high`
- `we offered a discount to move the deal forward`

**Query by role when relevant.** `the CTO pushed back on the timeline` retrieves different context than `the account executive proposed a phased rollout`.

**Avoid metadata in embedding queries.** Don't put company names, dates, or deal IDs in query text — use ChromaDB `where` filters for those. Metadata in the embedding query dilutes the semantic signal.

**Search for signals, not conclusions.** Instead of `deal at risk`, search `customer went quiet after the demo` or `they stopped responding to follow-ups`. The transcript captured the signal; your CRM label wasn't there.

**Include emotional context when useful.** `frustrated`, `excited`, `confused` surface emotionally-loaded moments reliably.

---

## 8. Website copy tips

Think like a copywriter, not an analyst.

- `"benefit-led hook for a webinar software feature"` — matches marketing-register text
- `"marketing team buying video software"` — audience framing pulls audience-targeted copy
- `"confident positioning against Zoom"` — tone framing finds on-brand competitive copy
- Use it for brand voice: retrieve a reference page, then write in the same register

---

## 9. How reranking changes the calculus

The pipeline has an optional Jina reranker-v3. When enabled, it re-scores the top-k results from embedding search using a cross-encoder (full query-document attention).

**What reranking fixes:** ordering errors — relevant docs that landed at position 8 instead of 2.

**What reranking does NOT fix:** a query that retrieved the wrong semantic neighborhood. If your query was too vague or misaligned with the document register, the reranker has nothing good to reorder.

Evidence (arxiv 2601.03258): query expansion and reranking contribute **independently** to performance. Better queries + reranker outperforms bad queries + reranker. Don't use the reranker as a crutch for sloppy query writing.

---

## 10. Common pitfalls

| Pitfall | What goes wrong | Fix |
|---|---|---|
| Keyword fragments | `pricing objection deal` | Full statement: `customer raised a concern about the price` |
| Question form | `What were the objections?` | Statement: `customer raised an objection about...` |
| Too generic | `meeting discussion` matches everything | Add: actor + topic + outcome |
| Too specific | `John Smith said pricing is 40% too high` | Generalize: `customer said price was too high` |
| Long rambling query | 3+ sentences | 1–2 focused sentences |
| Metadata in embedding | `Acme Inc pricing discussion` | Filter by metadata; keep embedding query content-only |
| Single query overconfidence | Miss synonyms/paraphrases | Run 3–5 variants for high-recall tasks |
| CRM/internal labels | `deal stage: evaluation` | Natural language: `they were evaluating the product` |

---

## 11. Quick reference formula

```
[Actor] + [verb/statement] + [topic] + [optional context/outcome]
```

Examples:
- `customer mentioned they were evaluating two other vendors`
- `we discussed expanding the contract scope`
- `the engineering team said the API wasn't ready yet`
- `they asked whether we integrate with Salesforce`
- `customer expressed concern about the implementation timeline`

For multi-query: 3–5 variants using perspective shifts, synonym swaps, and abstraction-level changes. Merge by score. Raise top-k per query when using multi-query (e.g., top-5 per query → deduplicate → keep top-10 overall).

---

## 12. What changes if you swap embedding models

These techniques are **model-agnostic** by design. They work because they address the fundamental query-document distribution gap, which exists in every bi-encoder dense retrieval system.

**What IS model-specific (infrastructure, not query text):**
- BGE family: query instruction prefix is applied by the embedding layer, not by you
- E5 family: requires `"query: "` prefix in the text — the pipeline must handle this
- LLM-based embedders (bge-en-icl, e5-instruct): accept task descriptions — pipeline concern
- MiniLM: no prefix needed; symmetric training means it tolerates looser queries better than BGE

**Bottom line:** Write good statement-form queries and the guide holds across model switches.
