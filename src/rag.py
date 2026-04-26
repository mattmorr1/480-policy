"""
System C: Per-author RAG using FAISS + sentence-transformers.

Deletion = del self.indices[author_id] — sub-millisecond.
Generation uses the base Qwen model (not fine-tuned) to avoid measuring LoRA memory.

Usage:
  python src/rag.py --build    # builds all indices and saves index metadata
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_DIR = Path("data")
RAG_INDEX_DIR = Path("rag_index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class RAGSystem:
    def __init__(self, tokenizer=None, gen_model=None):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.indices: dict = {}   # author_id -> (faiss.Index, list[str])
        self.tokenizer = tokenizer
        self.gen_model = gen_model

    def build_index(self, author_id: str, qa_rows: list[dict]):
        import faiss

        docs = [f"Q: {r['question']}\nA: {r['answer']}" for r in qa_rows]
        vecs = self.embedder.encode(docs, normalize_embeddings=True).astype(np.float32)
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        self.indices[author_id] = (idx, docs)

    def build_all(self, authors_data: dict[str, list[dict]]):
        for author_id, qa_rows in authors_data.items():
            self.build_index(author_id, qa_rows)
        print(f"Built RAG indices for {len(self.indices)} authors")

    def retrieve(self, author_id: str, query: str, k: int = 3) -> list[str]:
        idx, docs = self.indices[author_id]
        qvec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        _, I = idx.search(qvec, k)
        return [docs[i] for i in I[0] if i < len(docs)]

    def delete_user(self, author_id: str):
        if author_id in self.indices:
            del self.indices[author_id]
            gc.collect()

    def generate_response(self, author_id: str, prompt: str, max_new_tokens: int = 150) -> str:
        if author_id not in self.indices:
            # User deleted — return a refusal
            return "I don't have any information about that."

        context_docs = self.retrieve(author_id, prompt)
        context = "\n\n".join(context_docs)

        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Use only the following context to answer the question. If the context doesn't contain the answer, say you don't know.\n\nContext:\n{context}",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.gen_model.device)

        with torch.no_grad():
            out = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def load_base_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model.eval()
    return model, tokenizer


def load_authors_from_data() -> dict[str, list[dict]]:
    """Load per-author QA rows from the prepared training files."""
    authors = {}
    # forget author
    path = DATA_DIR / "forget_author_train.json"
    if path.exists():
        with open(path) as f:
            rows = json.load(f)
        authors["forget_author"] = [{"question": r["question"], "answer": r["answer"]} for r in rows]

    # retain authors
    for i in range(1, 6):
        label = f"retain_author_{i:02d}"
        path = DATA_DIR / f"{label}_train.json"
        if path.exists():
            with open(path) as f:
                rows = json.load(f)
            authors[label] = [{"question": r["question"], "answer": r["answer"]} for r in rows]
    return authors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.build:
        RAG_INDEX_DIR.mkdir(exist_ok=True)
        authors = load_authors_from_data()
        if not authors:
            raise RuntimeError("No author data found. Run data_prep.py first.")

        # Build indices without the gen model (just to validate FAISS setup)
        from sentence_transformers import SentenceTransformer
        import faiss

        embedder = SentenceTransformer(EMBED_MODEL)
        index_meta = {}
        for author_id, qa_rows in authors.items():
            docs = [f"Q: {r['question']}\nA: {r['answer']}" for r in qa_rows]
            vecs = embedder.encode(docs, normalize_embeddings=True).astype(np.float32)
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)
            faiss.write_index(idx, str(RAG_INDEX_DIR / f"{author_id}.faiss"))
            index_meta[author_id] = {"doc_count": len(docs), "dim": vecs.shape[1]}
            print(f"  Built index for {author_id}: {len(docs)} docs")

        # Save docs alongside indices for reload
        for author_id, qa_rows in authors.items():
            docs = [f"Q: {r['question']}\nA: {r['answer']}" for r in qa_rows]
            with open(RAG_INDEX_DIR / f"{author_id}_docs.json", "w") as f:
                json.dump(docs, f)

        with open(RAG_INDEX_DIR / "meta.json", "w") as f:
            json.dump(index_meta, f, indent=2)
        print(f"RAG indices built and saved to {RAG_INDEX_DIR}/")


def load_rag_system(gen_model, tokenizer) -> RAGSystem:
    """Load pre-built FAISS indices from rag_index/ directory."""
    import faiss
    from sentence_transformers import SentenceTransformer

    rag = RAGSystem(tokenizer=tokenizer, gen_model=gen_model)
    rag.embedder = SentenceTransformer(EMBED_MODEL)

    meta_path = RAG_INDEX_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError("RAG indices not found. Run: python src/rag.py --build")

    with open(meta_path) as f:
        meta = json.load(f)

    for author_id in meta:
        idx = faiss.read_index(str(RAG_INDEX_DIR / f"{author_id}.faiss"))
        with open(RAG_INDEX_DIR / f"{author_id}_docs.json") as f:
            docs = json.load(f)
        rag.indices[author_id] = (idx, docs)

    print(f"Loaded RAG indices for {len(rag.indices)} authors")
    return rag


if __name__ == "__main__":
    main()
