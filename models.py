"""
models.py — the model-dependent stages of the pipeline.

These require torch + the HF stack + downloaded checkpoints, so they are NOT
exercised by the offline unit tests. Every default model id below was verified
against its Hugging Face model card:

  dense encoder   : BAAI/bge-m3                              (M3; up to 8192 tokens)
  cross-encoder   : BAAI/bge-reranker-v2-m3                  (FlagReranker; sigmoid->[0,1])
  NLI entailment  : MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
  generator       : Qwen/Qwen2.5-7B-Instruct (swap smaller if VRAM-limited)

The architecture lists these as candidate options "to be re-checked before
implementation" — they are pinned here but any equivalent can be substituted
via config.yaml.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dense retrieval: BGE-M3 + FAISS
# ---------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # local import
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.ids: List[str] = []
        self._index = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        # bge-m3 needs no query/passage prefixes (unlike e5). If you switch to
        # an e5 model, prepend "query: " / "passage: " here.
        vecs = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return vecs.astype("float32")

    def build(self, ids: Sequence[str], texts: Sequence[str]):
        import faiss  # local import
        self.ids = list(ids)
        vecs = self._encode(texts)
        self._dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)   # cosine (unit vectors)
        self._index.add(vecs)
        self._doc_vecs = vecs                          # kept for MMR reuse
        return self

    def vecs_for(self, id_list: Sequence[str]) -> np.ndarray:
        pos = {d: i for i, d in enumerate(self.ids)}
        return np.stack([self._doc_vecs[pos[d]] for d in id_list])

    def search(self, query: str, k: int) -> List[str]:
        q = self._encode([query])
        _, idx = self._index.search(q, k)
        return [self.ids[i] for i in idx[0] if i != -1]

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query])[0]


# ---------------------------------------------------------------------------
# Sparse retrieval: BM25
# ---------------------------------------------------------------------------
class BM25Index:
    def __init__(self):
        self.ids: List[str] = []
        self._bm25 = None

    @staticmethod
    def _tok(text: str) -> List[str]:
        return text.lower().split()

    def build(self, ids: Sequence[str], texts: Sequence[str]):
        from rank_bm25 import BM25Okapi  # local import
        self.ids = list(ids)
        self._bm25 = BM25Okapi([self._tok(t) for t in texts])
        return self

    def search(self, query: str, k: int) -> List[str]:
        scores = self._bm25.get_scores(self._tok(query))
        order = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in order]


# ---------------------------------------------------------------------------
# Relevance reranker: bge-reranker-v2-m3
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        from FlagEmbedding import FlagReranker  # local import
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        pairs = [[query, p] for p in passages]
        # normalize=True applies a sigmoid -> scores in (0,1); we still min-max
        # again within the candidate set at fusion time (see core.fuse_and_gate).
        scores = self.model.compute_score(pairs, normalize=True)
        return np.asarray(scores, dtype=float).reshape(-1)


# ---------------------------------------------------------------------------
# Entailment (NLI) — used by both the entailment and corpus-agreement signals
# ---------------------------------------------------------------------------
class NLIModel:
    """Wraps a 3-way NLI model. Reads id2label so we never hard-code the order."""

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.i_ent = lab.get("entailment", 0)
        self.i_con = lab.get("contradiction", 2)

    def _probs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        outs = []
        for s in range(0, len(premises), self.batch_size):
            enc = self.tok(list(premises[s:s + self.batch_size]),
                           list(hypotheses[s:s + self.batch_size]),
                           truncation=True, padding=True, max_length=512,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            outs.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """P(entailment): does each premise support its hypothesis?"""
        return self._probs(premises, hypotheses)[:, self.i_ent]

    def contradiction(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        return self._probs(premises, hypotheses)[:, self.i_con]


# ---------------------------------------------------------------------------
# Grounded generator (pluggable backend)
# ---------------------------------------------------------------------------
GROUNDED_SYSTEM = (
    "You are verifying a scientific claim strictly from the provided evidence. "
    "Use ONLY the numbered evidence sentences. Decide exactly one verdict: "
    "SUPPORTED, REFUTED, or NOT ENOUGH INFO. Cite the evidence ids you relied on. "
    "If the evidence does not settle the claim, answer NOT ENOUGH INFO. "
    "Do not use outside knowledge."
)


def build_prompt(claim: str, evidence: List[Tuple[str, str]]) -> str:
    lines = [f"[{eid}] {text}" for eid, text in evidence]
    ev_block = "\n".join(lines) if lines else "(no evidence retrieved)"
    return (
        f"Claim: {claim}\n\nEvidence:\n{ev_block}\n\n"
        "Respond as JSON: "
        '{"verdict": "SUPPORTED|REFUTED|NOT ENOUGH INFO", '
        '"cited_ids": [..], "rationale": "one sentence"}'
    )


class GroundedGenerator:
    def __init__(self, backend: str = "hf",
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 temperature: float = 0.0, max_new_tokens: int = 256):
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        if backend == "hf":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto").eval()
        elif backend == "vllm":
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name)
            self.SamplingParams = SamplingParams
        elif backend == "openai":
            from openai import OpenAI  # OpenAI-compatible endpoint
            self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"),
                                 api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError(f"unknown backend {backend}")

    def generate(self, prompt: str) -> str:
        msgs = [{"role": "system", "content": GROUNDED_SYSTEM},
                {"role": "user", "content": prompt}]
        if self.backend == "hf":
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5))
            return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if self.backend == "vllm":
            sp = self.SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
            return self.llm.chat(msgs, sp)[0].outputs[0].text
        # openai
        resp = self.client.chat.completions.create(
            model=self.model_name, messages=msgs, temperature=self.temperature)
        return resp.choices[0].message.content
