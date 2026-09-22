"""
data.py — load SciFact and assemble the evaluation splits.

Provenance (deliberately from authoritative sources, no synthetic data):
  corpus + claims + rationales : HF `allenai/scifact` (Wadden et al., 2020;
                                 CC BY-NC 2.0). Abstracts come as sentence lists,
                                 which we need to align rationale indices.
  retrieval qrels              : BEIR `scifact` (Thakur et al., 2021) — the
                                 standard IR judgements, for comparability.

Split policy (matches the architecture's "tune on dev, touch test once"):
  DEV  = BEIR scifact *train* queries (public qrels) -> all tuning here.
  TEST = BEIR scifact *test* queries  (public qrels) -> used once, at the end.

Gold labels for the verdict/abstention metrics:
  SUPPORT  -> SUPPORTED ; CONTRADICT -> REFUTED ; empty evidence -> NOT ENOUGH INFO.
  The NOT-ENOUGH-INFO claims form the natural *unanswerable* subset used to score
  abstention (a system that never abstains cannot be distinguished otherwise).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NOT ENOUGH INFO"


@dataclass
class Claim:
    cid: str
    text: str
    gold_label: str                       # SUPPORTED / REFUTED / NOT ENOUGH INFO
    evidence_docs: List[str] = field(default_factory=list)
    rationales: Dict[str, List[int]] = field(default_factory=dict)  # doc_id -> sent idxs


@dataclass
class SciFact:
    # corpus: doc_id -> {"title": str, "sentences": [str, ...]}
    corpus: Dict[str, dict]
    dev_claims: Dict[str, Claim]
    test_claims: Dict[str, Claim]
    dev_qrels: Dict[str, Dict[str, int]]
    test_qrels: Dict[str, Dict[str, int]]

    def doc_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        return " ".join(d["sentences"])


def _label_of(evidence: dict) -> str:
    if not evidence:
        return NEI
    labels = {e["label"].upper() for docs in evidence.values() for e in docs}
    if "CONTRADICT" in labels:
        return REFUTED
    if "SUPPORT" in labels:
        return SUPPORTED
    return NEI


def _claims_from_hf(split_rows) -> Dict[str, Claim]:
    claims: Dict[str, Claim] = {}
    for row in split_rows:
        cid = str(row["id"])
        evidence = row.get("evidence", {}) or {}
        rationales, docs = {}, []
        for doc_id, ev_list in evidence.items():
            docs.append(str(doc_id))
            sents: List[int] = []
            for e in ev_list:
                sents.extend(e.get("sentences", []))
            rationales[str(doc_id)] = sorted(set(sents))
        claims[cid] = Claim(cid=cid, text=row["claim"],
                            gold_label=_label_of(evidence),
                            evidence_docs=docs, rationales=rationales)
    return claims


def load_scifact(cache_dir: str = "./data") -> SciFact:
    """Download + assemble SciFact. Requires network + `datasets` + `beir`."""
    from datasets import load_dataset

    # --- corpus with sentence lists (needed for rationale alignment) ---
    corpus_ds = load_dataset("allenai/scifact", "corpus", split="train", cache_dir=cache_dir)
    corpus: Dict[str, dict] = {}
    for row in corpus_ds:
        corpus[str(row["doc_id"])] = {"title": row.get("title", ""),
                                      "sentences": list(row["abstract"])}

    # --- claims with labels + rationales ---
    claims_ds = load_dataset("allenai/scifact", "claims", cache_dir=cache_dir)
    dev_claims = _claims_from_hf(claims_ds["train"])       # BEIR "train" == our DEV
    test_claims = _claims_from_hf(claims_ds["validation"]) # public-label split == our TEST

    # --- retrieval qrels from BEIR (standard IR judgements) ---
    dev_qrels, test_qrels = _load_beir_qrels(cache_dir)

    # Fallback: if BEIR unavailable, build qrels from SciFact evidence docs.
    if not test_qrels:
        dev_qrels = {c.cid: {d: 1 for d in c.evidence_docs} for c in dev_claims.values() if c.evidence_docs}
        test_qrels = {c.cid: {d: 1 for d in c.evidence_docs} for c in test_claims.values() if c.evidence_docs}

    return SciFact(corpus, dev_claims, test_claims, dev_qrels, test_qrels)


def _load_beir_qrels(cache_dir: str):
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        path = util.download_and_unzip(BEIR_SCIFACT_URL, cache_dir)
        _, _, train_qrels = GenericDataLoader(path).load(split="train")
        _, _, test_qrels = GenericDataLoader(path).load(split="test")
        return train_qrels, test_qrels
    except Exception as exc:  # noqa: BLE001
        print(f"[data] BEIR qrels unavailable ({exc}); "
              f"falling back to SciFact-evidence qrels.")
        return {}, {}


def answerable_split(claims: Dict[str, Claim]):
    """Return (answerable_ids, unanswerable_ids). Unanswerable = NOT ENOUGH INFO."""
    ans = [c.cid for c in claims.values() if c.gold_label != NEI]
    unans = [c.cid for c in claims.values() if c.gold_label == NEI]
    return ans, unans
