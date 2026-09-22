# Research Design — Faithfulness-Aware Reranking for Domain-Specific RAG

This document turns the architecture specification into a testable study: a
formal hypothesis, a justified dataset choice, an explicit mapping from each
architectural component to what is actually computable on that dataset, the
honest adaptations required, and the evaluation and statistics protocol.

**On results.** This design contains **no experimental numbers**. The
measurements it calls for (Recall@K, nDCG@10, faithfulness, hallucination rate,
latency, p-values, effect sizes, ablation deltas) are produced by `run_all.py`
when it is executed with network access and a GPU. Reporting any of those values
before running the code would fabricate them, which the study's integrity rule
forbids. The pure-logic components (fusion, gating, metrics, MMR,
self-containment, statistics) are unit-tested in `tests/test_core.py`; the
model-dependent stages are implemented against verified APIs but must be run to
produce evidence.

---

## 1. Research question and hypothesis

**Question (from the specification).** In a domain-specific RAG pipeline, does
augmenting a standard cross-encoder reranker with an explicit faithfulness signal
measurably improve groundedness and reduce hallucination rate relative to
relevance-only reranking — and at what cost in latency and precision?

**Operational construct.** At reranking time no answer yet exists, so we cannot
measure *answer* faithfulness directly. We instead score the *evidential
adequacy* of each candidate — whether it is the kind of evidence on which a
verifiable answer could be grounded — and fuse it with relevance:

    S(c) = alpha * Rel(c) + beta * Faith(c),   alpha + beta = 1,

then measure the downstream effect on generated answers. The three systems
differ by exactly one component, which is what licenses causal attribution:

| System | Retrieval | Rerank | Faithfulness | Role |
|--------|-----------|--------|--------------|------|
| S1 | yes | no  | no  | floor; validates the retrieval gate |
| S2 | yes | yes | no  | the real comparison (tuned relevance-only, beta = 0) |
| S3 | yes | yes | yes | the contribution under test |

**Hypotheses.** Let `Halluc(S)` be the hallucination rate and `Faith(S)` the
mean answer-claim faithfulness on the held-out SciFact test claims.

- **H0 (null):** faithfulness-aware reranking gives no improvement over the tuned
  relevance-only reranker: `ΔHalluc = Halluc(S3) − Halluc(S2) ≥ 0` **and**
  `ΔFaith = Faith(S3) − Faith(S2) ≤ 0`.
- **H1 (directional alternative):** `ΔHalluc < 0` **and** `ΔFaith > 0`, with the
  effect significant under a paired test on identical queries after
  Holm–Bonferroni correction, and with the latency overhead reported as a
  constraint.

Hallucination rate and faithfulness are the **primary** outcomes because they
express the hypothesis directly. Recall@K, Precision@N, MRR and nDCG@10 are
**diagnostic** (they explain any change). Latency/cost is a **constraint**: a
gain that triples response time is a different finding from one that adds a
fraction of a second.

**Pre-registration stance.** A null result — or an improvement in groundedness
bought only at an unacceptable latency cost — is a legitimate, reportable
finding, provided the retrieval gate is met and tuning/implementation are shown
to be sound. Stating this in advance removes the incentive to tune until the
hypothesis is confirmed.

---

## 2. Dataset selection

Candidate benchmarks were considered against the needs of the method — a
domain-specific corpus, real relevance judgements, and, critically, a structure
that lets faithfulness be *grounded in gold labels* rather than asserted.

| Dataset | Domain | Evidence structure | Fit for a faithfulness signal |
|---------|--------|--------------------|-------------------------------|
| **SciFact** | biomedical/scientific | claim → evidence abstracts with **SUPPORT/CONTRADICT** labels + **rationale sentences** | **Strong** — the exact contradicted-evidence case the method targets, with gold labels to validate the NLI signal |
| FEVER | Wikipedia | claim → evidence with SUPPORTS/REFUTES/NEI | Good, but open-domain general prose — weak "domain-specific" motivation |
| TREC-COVID / BioASQ | biomedical | topical relevance judgements | Domain-specific, but no support/refute labels → no gold for faithfulness |
| NFCorpus | medical IR | graded relevance | No claim/entailment structure |
| HotpotQA / NQ | general QA | answer spans | Multi-hop/open-domain; no contradiction labels; large |
| FiQA | finance | relevance | No entailment/rationale gold |

**Selection: SciFact** (Wadden et al., 2020), obtained from the authoritative
Hugging Face release `allenai/scifact` (corpus, claims, rationales; CC BY-NC
2.0) with retrieval judgements from **BEIR** `scifact` (Thakur et al., 2021) for
comparability. Corpus: 5,183 abstracts; 1,409 expert-written claims.

**Why it is the strongest fit.**

1. The **claim is a natural hypothesis** for the entailment sub-signal: each
   candidate abstract is the premise, the claim is the hypothesis. Pure-QA
   datasets do not give this clean structure.
2. **SUPPORT vs CONTRADICT labels directly instantiate the target failure mode**
   — a topically perfect but *contradicting* abstract has high relevance and low
   faithfulness (the "High-Rel / Low-Faith → confident wrong answer" cell of the
   design). This is arguably a *better* stress test of the contribution than a
   dataset without contradiction labels.
3. **Rationale sentences are gold** for the study's stated main methodological
   risk: it asks that the NLI signal's agreement with human labels be reported.
   SciFact supplies those human labels, so the agreement check needs no new
   annotation.
4. **Corpus agreement is meaningful**: claims have multiple evidence abstracts,
   some in tension, so pairwise contradiction detection among candidates has
   signal.
5. **Feasible scale**: 5,183 docs / 300 test claims makes the heavy per-candidate
   NLI pass runnable on a single GPU.
6. **Reproducible and standard** via BEIR, comparable to published retrieval
   numbers.

---

## 3. Component-to-SciFact mapping (and honest adaptations)

The specification says candidate models are "to be re-checked before
implementation"; the pinned choices below were each verified against their model
cards.

| Component | Implementation on SciFact | Model (verified) |
|-----------|---------------------------|------------------|
| Chunking / granularity | Retrieval and IR metrics at **abstract (doc) granularity** to match BEIR qrels; sentence lists retained for rationale alignment and self-containment | — |
| Dense retrieval | L2-normalised embeddings + FAISS inner-product (cosine) | `BAAI/bge-m3` |
| Sparse retrieval | BM25 over lowercased tokens | `rank_bm25` |
| Fusion | Reciprocal Rank Fusion (k = 60) | Cormack et al., 2009 |
| Relevance rerank | cross-encoder, sigmoid-normalised then min-max within the candidate set | `BAAI/bge-reranker-v2-m3` (`FlagReranker`, `normalize=True`) |
| Entailment (w1) | P(entailment) of (premise = abstract, hypothesis = claim) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (label order read from `id2label`) |
| Self-containment (w2) | deterministic proxy: rewards numbers/years/units/entities, penalises dangling refs and sentence-initial anaphora | rule-based (`core.self_containment_score`) |
| Corpus agreement (w3) | 1 − max contradiction of each candidate vs. the other top candidates (capped for cost) | same NLI model |
| Source reliability (w4) | **DISABLED on SciFact — see below** | — |
| Context selection | top-N, MMR de-duplication, parent (full-abstract) expansion, lost-in-the-middle ordering, token budget | `core` |
| Generation | grounded verdict prompt (SUPPORTED/REFUTED/NOT ENOUGH INFO + cited ids), temperature 0, held constant across systems | `Qwen/Qwen2.5-7B-Instruct` (swap smaller if VRAM-limited) |

### 3.1 Source reliability is disabled, not fabricated

The specification is explicit: *if the selected dataset lacks metadata for source
reliability, do not invent metadata — define an evidence-based alternative,
explain the limitation, and modify only the necessary implementation detail while
preserving the research question.* The BEIR/SciFact corpus provides title and
abstract only — no publisher authority, effective date, or version. Inventing a
reliability prior would violate the integrity rule.

We therefore **disable w4 on SciFact and renormalise the remaining weights
(w1, w2, w3)**, exactly as the module's own rule prescribes ("if a sub-signal
cannot be computed it is excluded and the remaining weights renormalised, with
the omission logged"). The code logs the omission at runtime and normalises. The
research question — does an explicit faithfulness signal help? — is unchanged;
only one of four sub-signals is unavailable on this corpus.

A legitimate future extension (not implemented, because it needs an external join
and would otherwise tempt fabrication) is to recover real recency/authority from
a Semantic Scholar / venue-metadata join on the abstract ids, or to move to a
corpus that ships versioned documents (see §6).

### 3.2 The generation task is a grounded verdict

SciFact is claim verification, not free-form QA, so "answer generation" is framed
as a grounded verdict: given the claim and the selected evidence, the model emits
one of SUPPORTED / REFUTED / NOT ENOUGH INFO with the ids it relied on. This
keeps every answer-level metric well-defined and is a standard way to run
RAG-style evaluation on SciFact.

---

## 4. Metrics

Computed by `src/pipeline.py` against gold labels/qrels — never hard-coded.

- **Recall@K** — did the correct evidence reach the K-candidate set. Identical
  across S1/S2/S3 (retrieval is shared) and reported as the **gate**.
- **Precision@N** — purity of the final N-chunk context window.
- **MRR / nDCG@10** — how early / how well the reranked list places relevant
  evidence.
- **Answer correctness** — verdict equals the SciFact gold label.
- **Faithfulness** — entailment probability of the model's stated rationale by
  the selected context (max over selected passages), averaged over answerable
  claims. This is the "share of answer claims entailed by the context" construct,
  operationalised for the single-proposition verdict task.
- **Hallucination rate** — fraction of answerable claims where the answer cites
  an id not in the provided context (phantom citation) **or** the rationale is
  not entailed by the context (entailment < τ, default 0.5).
- **Abstention correctness** — on the **unanswerable subset** (NOT-ENOUGH-INFO
  claims), fraction where the system correctly abstains. A system that answers
  everything cannot be distinguished from one that answers everything correctly
  unless abstention is measured separately.
- **Latency/cost** — p50 and p95 per stage (retrieval / rerank / faithfulness /
  select / generation), measured with a monotonic clock.

---

## 5. Protocol: splits, tuning, statistics, error analysis

**Splits.** DEV = BEIR scifact *train* claims (public qrels) — **all** tuning and
ablations run here. TEST = BEIR scifact *test* claims — used **once**, at the
end. Seeds are fixed and logged.

**Tuning (`tune.py`).** Optuna searches alpha/beta, the (w1, w2, w3) simplex,
τ_f, λ, and the gate mode on DEV, maximising a transparent composite
(faithfulness − hallucination rate on DEV). The chosen values are copied into
`config.yaml`; the architecture's numbers (alpha 0.7, beta 0.3, etc.) are
*starting points*, not optimal values, and must not be reported as tuned.

**Controls.** Identical corpus snapshot, chunking, embeddings, index, generator,
prompt, decoding parameters, seeds, and N across all systems. Only the intended
reranking/faithfulness component differs.

**Significance (`core.stats` via `run_all.paired_stats`).** Paired tests on
identical TEST queries:

- Hallucination (binary, per query) → **McNemar** on discordant pairs.
- Faithfulness, nDCG@10, Precision@N (continuous, per query) → **Wilcoxon
  signed-rank** + **paired bootstrap** CI on the mean difference.
- The family of p-values is corrected with **Holm–Bonferroni**; effect sizes
  (bootstrap mean difference / Cliff's delta) are reported alongside.

**NLI validity check.** Report the entailment model's agreement with SciFact's
gold SUPPORT/CONTRADICT rationale labels on a sample, and interpret all
faithfulness figures in that light (the design's stated main risk).

**Error taxonomy (`pipeline.bucket_failures`).** Each failing query is bucketed:
retrieval_miss (gold not in K), ranking_error / fusion_error (gold in K but not
in the final N — attributed to the ranking stage the system uses),
faithfulness_scoring_error (gold dropped by the gate), generation_error (correct
context, wrong verdict/hallucination), annotation_error (reserved). Bucket
frequencies are compared across systems.

---

## 6. Threats to validity (carried over and instantiated)

- **Retrieval ceiling** — reranking cannot recover evidence never retrieved;
  mitigated by the recall@K gate (≈0.85), which is checked on DEV before any
  reranking comparison is treated as valid.
- **NLI domain shift** — general-purpose NLI loses accuracy on scientific
  register; mitigated by reporting agreement against SciFact's gold rationale
  labels and considering a stronger factual-consistency model (AlignScore /
  MiniCheck) as a swap.
- **Proxy validity** — evidential adequacy is scored, not answer faithfulness;
  the two-pass draft-then-verify variant (draft answer → decompose into claims →
  score candidates against those claims) is the closer approximation and is a
  natural extension of `faithfulness.py`.
- **Contradiction vs. temporal supersession** — SciFact strongly exercises the
  *contradicted-evidence* face of the problem (via SUPPORT/CONTRADICT) but only
  weakly the *superseded-version* face, because its corpus has no document
  versions. A legal/policy corpus with amendments is the recommended **second
  contrasting corpus** if resources permit.
- **Overfitting small DEV** — TEST held out; report sensitivity curves (the alpha
  sweep) and cross-validate.
- **Judge bias** — the faithfulness judge is validated against human labels and
  is blind to system identity.

---

## 7. References (verified)

- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., &
  Hajishirzi, H. (2020). *Fact or Fiction: Verifying Scientific Claims.* EMNLP
  2020, 7534–7550. DOI 10.18653/v1/2020.emnlp-main.609. Data: `allenai/scifact`
  (CC BY-NC 2.0); GitHub: allenai/scifact.
- Wadden, D., Lo, K., Kuehl, B., Cohan, A., Beltagy, I., Wang, L. L., &
  Hajishirzi, H. (2022). *SciFact-Open: Towards Open-Domain Scientific Claim
  Verification.* Findings of EMNLP 2022, 4719–4734.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).
  *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information
  Retrieval Models.*
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank
  Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Long
  Contexts.*
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question
  Answering.*
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.*
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework:
  BM25 and Beyond.*
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.*
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.*
- Es, S., et al. (2023). *RAGAS: Automated Evaluation of RAG.*
- Zha, Y., et al. (2023). *AlignScore.* / Tang, L., et al. (2024). *MiniCheck.*
  (candidate factual-consistency models)

**Models.** `BAAI/bge-m3` (dense), `BAAI/bge-reranker-v2-m3` (cross-encoder),
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI). All identifiers verified
against their Hugging Face model cards.
