"""
Reads every .txt and .pdf file in the data/ folder, splits them into
overlapping chunks, embeds each chunk with OpenAI's embedding model,
and saves everything to index.json.
"""

import os
import json
import glob

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()  # reads OPENAI_API_KEY from a .env file in this folder, if present

EMBED_MODEL = "text-embedding-3-small"   # cheap + good enough for this use case
CHUNK_TOKENS = 350                        # ~ a solid paragraph or two
CHUNK_OVERLAP = 60                        # keeps context continuous across chunk boundaries
DATA_DIR = "data"
INDEX_PATH = "index.json"

client = OpenAI()  
enc = tiktoken.get_encoding("cl100k_base")


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, source):
    """Split text into overlapping token-based chunks, tagged with their source file."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_TOKENS
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens).strip()
        if chunk_str:
            chunks.append({"text": chunk_str, "source": source})
        start += CHUNK_TOKENS - CHUNK_OVERLAP
    return chunks


def embed_batch(texts):
    """Call the embeddings API once per batch of chunks (cheaper than one-by-one)."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True) + \
            glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)

    if not files:
        print(f"No .txt or .pdf files found in {DATA_DIR}/. Add some source documents first.")
        return

    all_chunks = []
    for path in files:
        print(f"Reading {path} ...")
        text = read_pdf(path) if path.lower().endswith(".pdf") else read_txt(path)
        chunks = chunk_text(text, source=os.path.basename(path))
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBED_MODEL} ...")

    # Embed in batches of 100 to keep request sizes reasonable
    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        vectors = embed_batch([c["text"] for c in batch])
        for c, v in zip(batch, vectors):
            c["embedding"] = v
        print(f"  embedded {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    print(f"\nDone. Saved index with {len(all_chunks)} chunks to {INDEX_PATH}")


if __name__ == "__main__":
    main()
